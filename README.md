CUDA optimization of SIFT
ECE569: High Performance Computing

### How to build project (from an interactive session on the Ocelote cluster)
```
module load cuda11/11.0
cd sift_parallel
mkdir build
cd build
cmake ..
make
```

### How to run on a single image (assumes you are in root project directory `cuda_project`)
```
cd sift_parallel/bin
./serial_and_parallel ../imgs/ParallelTestData/cat.jpg
```

### References:
Built off of a serial implementation of SIFT by dbarac: https://github.com/dbarac/sift-cpp/tree/master
