CUDA optimization of SIFT

ECE569: High Performance Computing

-------------------------------------------------------

## How to build parallel code (from an interactive session on the Ocelote cluster)
##### (assumes your current directory is root project directory: `cuda_project`)
```
module load cuda11/11.0
cd sift_parallel
mkdir build
cd build
cmake ..
make
```

## How to run parallel code 
##### (assumes your current directory is root project directory: `cuda_project`)

### Running executable to compare serial, naive parallel, and optimized parallel code
```
cd sift_parallel/bin
./compare_serial_parallel ../imgs/ParallelTestData/cat.jpg
```

### Producing an image with features using optimized parallel code
will create/overwrite a `result.jpg` file in current directory with features drawn onto input image
```
cd sift_parallel/bin
./find_keypoints ../imgs/ParallelTestData/cat.jpg
```


### Matching features across images using optimized parallel code
will create/overwrite a `result.jpg` file in current directory with visualized feature matching across images
```
cd sift_parallel/bin
./match_features ../imgs/book.png ../imgs/book_in_scene.png
```

-------------------------------------------------------------

## How to build serial code 
##### (assumes your current directory is root project directory `cuda_project`)
```
cd sift_serial
mkdir build
cd build
cmake ..
make
```

## How to run serial code
##### (assumes your current directory is root project directory `cuda_project`)

### Producing an image with features with serial code
will create/overwrite a `result.jpg` file in current directory with features drawn onto input image
```
cd sift_serial/bin
./find_keypoints ../imgs/book.png
```

### Matching features across images using serial code
will create/overwrite a `result.jpg` file in current directory with visualized feature matching across images
```
cd sift_serial/bin
./find_keypoints ../imgs/book.png ../imgs/book_in_scene.png
```

----------------------------------------------

### References:
Built off of a serial implementation of SIFT by dbarac: https://github.com/dbarac/sift-cpp/tree/master
