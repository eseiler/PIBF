// -----------------------------------------------------------------------------------------------------
// Copyright (c) 2006-2019, Knut Reinert & Freie Universität Berlin
// Copyright (c) 2016-2019, Knut Reinert & MPI für molekulare Genetik
// This file may be used, modified and/or redistributed under the terms of the 3-clause BSD-License
// shipped with this file and also available at: https://github.com/seqan/seqan3/blob/master/LICENSE
// -----------------------------------------------------------------------------------------------------

#include <random>
#include <benchmark/benchmark.h>
#include <seqan/binning_directory.h>
#include <atomic>

using namespace seqan;

CharString baseDir{std::string{{BASE_DIR}} + "/example_data/"};
uint64_t e{3};

template <typename TValue, typename THash, typename TBitvector, typename TChunks>
static void insertKmer_IBF(benchmark::State& state)
{
    auto bins = state.range(0);
    auto k = state.range(1);
    auto bits = state.range(2);
    auto hash = state.range(3);
    auto threads = state.range(4);
    BinningDirectory<InterleavedBloomFilter, BDConfig<TValue, THash, TBitvector, TChunks> > ibf (bins, hash, k, (1ULL<<bits));

    for (auto _ : state)
    {
        insertKmerDir(ibf, toCString(baseDir), threads);
    }

    CharString storage("");
    append(storage, CharString(std::to_string(bins)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(k)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(bits)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(TChunks::VALUE)));
    if constexpr (std::is_same_v<TBitvector, Uncompressed>)
    {
        append(storage, CharString("_Uncompressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, Compressed>)
    {
        append(storage, CharString("_Compressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, CompressedDisk>)
    {
        append(storage, CharString("_CompressedDisk"));
    }
    else
    {
        append(storage, CharString("_UncompressedDisk"));
    }
    append(storage, CharString("_ibf.filter"));

    store(ibf, storage);

    state.counters["Size"] = size(ibf);
}

template <typename TValue, typename THash, typename TBitvector, typename TChunks>
static void select_IBF(benchmark::State& state)
{
    auto bins = state.range(0);
    auto k = state.range(1);
    auto bits = state.range(2);
    auto threads = state.range(4);

    CharString storage("");
    append(storage, CharString(std::to_string(bins)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(k)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(bits)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(TChunks::VALUE)));
    if constexpr (std::is_same_v<TBitvector, Uncompressed>)
    {
        append(storage, CharString("_Uncompressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, Compressed>)
    {
        append(storage, CharString("_Compressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, CompressedDisk>)
    {
        append(storage, CharString("_CompressedDisk"));
    }
    else
    {
        append(storage, CharString("_UncompressedDisk"));
    }
    append(storage, CharString("_ibf.filter"));

    auto fullTime = std::chrono::high_resolution_clock::now();

    double loadingTime{0.0};
    auto start = std::chrono::high_resolution_clock::now();
    BinningDirectory<InterleavedBloomFilter, BDConfig<TValue, THash, TBitvector> > ibf (storage);
    auto end   = std::chrono::high_resolution_clock::now();
    loadingTime = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();


    std::atomic<uint64_t> verifications{0};
    std::atomic<uint64_t> tp{0};
    std::atomic<uint64_t> p{0};
    std::atomic<uint64_t> fp{0};
    std::atomic<uint64_t> fn{0};
    std::atomic<uint64_t> readNo{0};

    for (auto _ : state)
    {
        double selectTime{0.0};
        double ioTime{0.0};
        Semaphore thread_limiter(threads);
        std::mutex mtx;
        std::mutex mtx2;
        std::vector<std::future<void>> tasks;

        for(int32_t i = 0; i < bins; ++i)
        {
            CharString file(baseDir);
            append(file, CharString(std::to_string(bins)));
            append(file, CharString{"/reads/bin_"});
            append(file, CharString(std::string(numDigits(bins)-numDigits(i), '0') + (std::to_string(i))));
            append(file, CharString(".fastq"));

            tasks.emplace_back(
                std::async(std::launch::async, [&, file, i] {
                    Critical_section _(thread_limiter);
                    CharString id;
                    String<TValue> seq;
                    SeqFileIn seqFileIn;
                    uint64_t c{0};
                    if (!open(seqFileIn, toCString(file)))
                    {
                        CharString msg = "Unable to open contigs file: ";
                        append(msg, CharString(file));
                        throw toCString(msg);
                    }
                    while(!atEnd(seqFileIn))
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        readRecord(id, seq, seqFileIn);
                        auto end   = std::chrono::high_resolution_clock::now();
                        mtx2.lock();
                        ioTime += std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
                        mtx2.unlock();

                        start = std::chrono::high_resolution_clock::now();
                        auto res = select(ibf, seq, e);
                        end   = std::chrono::high_resolution_clock::now();
                        ++readNo;
                        mtx.lock();
                        selectTime += std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
                        mtx.unlock();

                        if (res[i])
                            ++tp;
                        else
                            ++fn;
                        c = count(res.begin(), res.end(), true);
                        verifications += c;
                        if (c > 1)
                        {
                            if (res[i])
                                fp += c - 1;
                            else
                                fp += c;
                        }
                        p += c;
                    }
                })
            );
        }
        for (auto &&task : tasks){
            task.get();
        }

        auto fullTime2   = std::chrono::high_resolution_clock::now();
        state.counters["TP"] = tp.load();
        state.counters["FN"] = fn.load();
        state.counters["FP"] = fp.load();
        state.counters["P"] = p.load();
        state.counters["readNo"] = readNo.load();
        state.counters["verifications"] = verifications.load();
        state.counters["Verifications"] = static_cast<double>(verifications.load())/readNo.load();
        state.counters["Sensitivity"] = static_cast<double>(tp.load())/readNo.load();
        state.counters["Precision"] = static_cast<double>(tp.load())/p.load();
        state.counters["FNR"] = static_cast<double>(fn.load())/readNo.load();
        state.counters["FDR"] = static_cast<double>(fp.load())/p.load();
        state.counters["loadingTime"] = loadingTime;
        state.counters["ioTime"] = ioTime;
        state.counters["selectTime"] = selectTime;
        state.counters["vectorSize"] = size(ibf);
        state.counters["fullTime"] = std::chrono::duration_cast<std::chrono::duration<double> >(fullTime2 - fullTime).count();
    }
}

template <typename TValue, typename THash, typename TBitvector, typename TChunks>
static void select_IBFChunked(benchmark::State& state)
{
    auto bins = state.range(0);
    auto k = state.range(1);
    auto bits = state.range(2);
    auto threads = state.range(4);
    auto chunks = TChunks::VALUE;

    CharString storage("");
    append(storage, CharString(std::to_string(bins)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(k)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(bits)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(TChunks::VALUE)));
    if constexpr (std::is_same_v<TBitvector, Uncompressed>)
    {
        append(storage, CharString("_Uncompressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, Compressed>)
    {
        append(storage, CharString("_Compressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, CompressedDisk>)
    {
        append(storage, CharString("_CompressedDisk"));
    }
    else
    {
        append(storage, CharString("_UncompressedDisk"));
    }
    append(storage, CharString("_ibf.filter"));

    auto fullTime = std::chrono::high_resolution_clock::now();

    double loadingTime{0.0};
    auto start = std::chrono::high_resolution_clock::now();
    BinningDirectory<InterleavedBloomFilter, BDConfig<TValue, THash, TBitvector, TChunks> > ibf (storage);
    auto end   = std::chrono::high_resolution_clock::now();
    loadingTime = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

    std::atomic<uint64_t> verifications{0};
    std::atomic<uint64_t> tp{0};
    std::atomic<uint64_t> p{0};
    std::atomic<uint64_t> fp{0};
    std::atomic<uint64_t> fn{0};
    std::atomic<uint64_t> readNo{0};

    std::vector<std::vector<uint64_t>> countTemp;
    countTemp.resize(1ULL<<16, std::vector<uint64_t>(bins));
    for (auto _ : state)
    {
        double selectTime{0.0};
        double ioTime{0.0};
        Semaphore thread_limiter(threads);
        std::mutex mtx;
        std::mutex mtx2;
        std::vector<std::future<void>> tasks;

        for (uint8_t chunk = 0; chunk < chunks; ++chunk)
        {
            tasks.clear();
            for(int32_t i = 0; i < bins; ++i)
            {
                CharString file(baseDir);
                append(file, CharString(std::to_string(bins)));
                append(file, CharString{"/reads/bin_"});
                append(file, CharString(std::string(numDigits(bins)-numDigits(i), '0') + (std::to_string(i))));
                append(file, CharString(".fastq"));

                tasks.emplace_back(
                    std::async(std::launch::async, [&, file, i] {
                        Critical_section _(thread_limiter);
                        CharString id;
                        String<TValue> seq;
                        uint32_t id_i;
                        SeqFileIn seqFileIn;
                        if (!open(seqFileIn, toCString(file)))
                        {
                            CharString msg = "Unable to open contigs file: ";
                            append(msg, CharString(file));
                            throw toCString(msg);
                        }
                        while(!atEnd(seqFileIn))
                        {
                            auto start = std::chrono::high_resolution_clock::now();
                            readRecord(id, seq, seqFileIn);
                            id_i = static_cast<uint32_t>(std::atoi(toCString(id)));
                            auto end   = std::chrono::high_resolution_clock::now();
                            mtx2.lock();
                            ioTime += std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
                            mtx2.unlock();

                            start = std::chrono::high_resolution_clock::now();
                            // auto res = select(ibf, seq, e);
                            std::vector<uint64_t> res = count(ibf, seq, 3u, chunk);
                            std::transform(countTemp[id_i].begin(), countTemp[id_i].end(), res.begin(), countTemp[id_i].begin(), std::plus<uint64_t>());
                            end   = std::chrono::high_resolution_clock::now();
                            ++readNo;
                            mtx.lock();
                            selectTime += std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
                            mtx.unlock();

                        }
                    })
                );
            }
            for (auto &&task : tasks){
                task.get();
            }
        }
        uint64_t threshold{25}; //100-(2+1)*19+1
        uint64_t span = (1ULL<<16) / bins;
        for (uint64_t r = 0; r < (1ULL<<16); ++r)
        {
            uint64_t i = r / span;
            if (countTemp[r][i] >= threshold)
                ++tp;
            else
                ++fn;
            uint64_t c = std::count_if(countTemp[r].begin(), countTemp[r].end(), [threshold](uint64_t j){return j >= threshold;});
            verifications += c;
            if (c > 1)
            {
                if (countTemp[r][i] >= threshold)
                    fp += c - 1;
                else
                    fp += c;
            }
            p += c;
        }

        auto fullTime2   = std::chrono::high_resolution_clock::now();
        uint64_t readNoN = readNo.load() / chunks;
        state.counters["TP"] = tp.load();
        state.counters["FN"] = fn.load();
        state.counters["FP"] = fp.load();
        state.counters["P"] = p.load();
        state.counters["readNo"] = readNoN;
        state.counters["verifications"] = verifications.load();
        state.counters["Verifications"] = static_cast<double>(verifications.load())/readNoN;
        state.counters["Sensitivity"] = static_cast<double>(tp.load())/readNoN;
        state.counters["Precision"] = static_cast<double>(tp.load())/p.load();
        state.counters["FNR"] = static_cast<double>(fn.load())/readNoN;
        state.counters["FDR"] = static_cast<double>(fp.load())/p.load();
        state.counters["loadingTime"] = loadingTime;
        state.counters["ioTime"] = ioTime;
        state.counters["selectTime"] = selectTime;
        state.counters["vectorSize"] = size(ibf);
        state.counters["fullTime"] = std::chrono::duration_cast<std::chrono::duration<double> >(fullTime2 - fullTime).count();
    }
}

template <typename TValue, typename THash, typename TBitvector, typename TChunks>
static void insertKmer_DA(benchmark::State& state)
{
    auto bins = state.range(0);
    auto k = state.range(1);
    auto threads = state.range(2);
    BinningDirectory<DirectAddressing, BDConfig<TValue, THash, TBitvector, TChunks> > da (bins, k);

    for (auto _ : state)
    {
        insertKmerDir(da, toCString(baseDir), threads);
    }

    CharString storage("");
    append(storage, CharString(std::to_string(bins)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(k)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(TChunks::VALUE)));
    if constexpr (std::is_same_v<TBitvector, Uncompressed>)
    {
        append(storage, CharString("_Uncompressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, Compressed>)
    {
        append(storage, CharString("_Compressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, CompressedDisk>)
    {
        append(storage, CharString("_CompressedDisk"));
    }
    else
    {
        append(storage, CharString("_UncompressedDisk"));
    }
    append(storage, CharString("_da.filter"));
    store(da, storage);

    state.counters["Size"] = size(da);
}

template <typename TValue, typename THash, typename TBitvector, typename TChunks>
static void select_DA(benchmark::State& state)
{
    auto bins = state.range(0);
    auto k = state.range(1);
    auto threads = state.range(2);

    CharString storage("");
    append(storage, CharString(std::to_string(bins)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(k)));
    append(storage, CharString("_"));
    append(storage, CharString(std::to_string(TChunks::VALUE)));
    if constexpr (std::is_same_v<TBitvector, Uncompressed>)
    {
        append(storage, CharString("_Uncompressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, Compressed>)
    {
        append(storage, CharString("_Compressed"));
    }
    else if constexpr (std::is_same_v<TBitvector, CompressedDisk>)
    {
        append(storage, CharString("_CompressedDisk"));
    }
    else
    {
        append(storage, CharString("_UncompressedDisk"));
    }
    append(storage, CharString("_da.filter"));

    auto fullTime = std::chrono::high_resolution_clock::now();

    double loadingTime{0.0};
    auto start = std::chrono::high_resolution_clock::now();
    BinningDirectory<DirectAddressing, BDConfig<TValue, THash, TBitvector, TChunks> > da (storage);
    auto end   = std::chrono::high_resolution_clock::now();
    loadingTime = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

    std::atomic<uint64_t> verifications{0};
    std::atomic<uint64_t> tp{0};
    std::atomic<uint64_t> p{0};
    std::atomic<uint64_t> fp{0};
    std::atomic<uint64_t> fn{0};
    std::atomic<uint64_t> readNo{0};

    for (auto _ : state)
    {
        double selectTime{0.0};
        double ioTime{0.0};
        Semaphore thread_limiter(threads);
        std::mutex mtx;
        std::mutex mtx2;
        std::vector<std::future<void>> tasks;

        for(int32_t i = 0; i < bins; ++i)
        {
            CharString file(baseDir);
            append(file, CharString(std::to_string(bins)));
            append(file, CharString{"/reads/bin_"});
            append(file, CharString(std::string(numDigits(bins)-numDigits(i), '0') + (std::to_string(i))));
            append(file, CharString(".fastq"));

            tasks.emplace_back(
                std::async(std::launch::async, [&, file, i] {
                    Critical_section _(thread_limiter);
                    CharString id;
                    String<TValue> seq;
                    SeqFileIn seqFileIn;
                    uint64_t c{0};
                    if (!open(seqFileIn, toCString(file)))
                    {
                        CharString msg = "Unable to open contigs file: ";
                        append(msg, CharString(file));
                        throw toCString(msg);
                    }
                    while(!atEnd(seqFileIn))
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        readRecord(id, seq, seqFileIn);
                        auto end   = std::chrono::high_resolution_clock::now();
                        mtx2.lock();
                        ioTime += std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
                        mtx2.unlock();

                        start = std::chrono::high_resolution_clock::now();
                        auto res = select(da, seq, e);
                        end   = std::chrono::high_resolution_clock::now();
                        ++readNo;
                        mtx.lock();
                        selectTime += std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
                        mtx.unlock();

                        if (res[i])
                            ++tp;
                        else
                            ++fn;
                        c = count(res.begin(), res.end(), true);
                        verifications += c;
                        if (c > 1)
                        {
                            if (res[i])
                                fp += c - 1;
                            else
                                fp += c;
                        }
                        p += c;
                    }
                })
            );
        }
        for (auto &&task : tasks){
            task.get();
        }

        auto fullTime2   = std::chrono::high_resolution_clock::now();
        state.counters["TP"] = tp.load();
        state.counters["FN"] = fn.load();
        state.counters["FP"] = fp.load();
        state.counters["P"] = p.load();
        state.counters["readNo"] = readNo.load();
        state.counters["verifications"] = verifications.load();
        state.counters["Verifications"] = static_cast<double>(verifications.load())/readNo.load();
        state.counters["Sensitivity"] = static_cast<double>(tp.load())/readNo.load();
        state.counters["Precision"] = static_cast<double>(tp.load())/p.load();
        state.counters["FNR"] = static_cast<double>(fn.load())/readNo.load();
        state.counters["FDR"] = static_cast<double>(fp.load())/p.load();
        state.counters["loadingTime"] = loadingTime;
        state.counters["ioTime"] = ioTime;
        state.counters["selectTime"] = selectTime;
        state.counters["vectorSize"] = size(da);
        state.counters["fullTime"] = std::chrono::duration_cast<std::chrono::duration<double> >(fullTime2 - fullTime).count();
    }
}

[[maybe_unused]]
static void IBFArguments(benchmark::internal::Benchmark* b)
{
    int32_t binNo{256};
    int32_t threads{1};
    for (int32_t k = 19; k < 20; ++k)
    {
        for (int32_t bits = 23; bits <= 25; ++bits)
        {
            for (int32_t hashNo = 3; hashNo < 4; ++hashNo)
            {
                b->Args({binNo, k, bits, hashNo, threads});
            }
        }
    }
}

[[maybe_unused]]
static void DAArguments(benchmark::internal::Benchmark* b)
{
    int32_t binNo{256};
    int32_t threads{1};
    for (int32_t k = 7; k <= 8; ++k)
    {
        b->Args({binNo, k, threads});
    }
}

BENCHMARK_TEMPLATE(insertKmer_IBF, Dna, Normal<5>, Uncompressed, Chunks<1>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(insertKmer_IBF, Dna, Normal<5>, Compressed, Chunks<1>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(insertKmer_IBF, Dna, Normal<5>, UncompressedDisk, Chunks<4>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(insertKmer_IBF, Dna, Normal<5>, CompressedDisk, Chunks<4>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(insertKmer_DA, Dna, Normal<5>, Uncompressed, Chunks<1>)->Apply(DAArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(insertKmer_DA, Dna, Normal<5>, Compressed, Chunks<1>)->Apply(DAArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);

BENCHMARK_TEMPLATE(select_IBF, Dna, Normal<5>, Uncompressed, Chunks<1>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(select_IBF, Dna, Normal<5>, Compressed, Chunks<1>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(select_IBFChunked, Dna, Normal<5>, UncompressedDisk, Chunks<4>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(select_IBFChunked, Dna, Normal<5>, CompressedDisk, Chunks<4>)->Apply(IBFArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(select_DA, Dna, Normal<5>, Uncompressed, Chunks<1>)->Apply(DAArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);
BENCHMARK_TEMPLATE(select_DA, Dna, Normal<5>, Compressed, Chunks<1>)->Apply(DAArguments)->Unit(benchmark::kMillisecond)->MinTime(0.000000001);

BENCHMARK_MAIN();
