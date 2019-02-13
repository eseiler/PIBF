# PIBF
Partitioned Interleaved Bloom Filter


## Requirements
* cmake > 3.2
* gcc >= 7

## Example
Replace `g++-7` by the executable/path to your gcc>=7 `g++` executable.

```bash
git clone --recursive https://github.com/eseiler/PIBF
cd PIBF
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-7
make
./example
```
