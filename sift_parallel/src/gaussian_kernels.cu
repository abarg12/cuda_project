// CUDA kernels for constructing Guassian, DoG, and Gradient Pyramids
#include "sift.hpp"


// Row convolution
__global__ void gaussianBlurRow(
    const float* input, float* output,
    int width, int height,
    const float* kernel, int kSize, int kCenter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int k = 0; k < kSize; ++k) {
        int dx = x + (k - kCenter);
        if (dx < 0) dx = 0;
        if (dx >= width) dx = width - 1;
        sum += input[y * width + dx] * kernel[k];
    }
    output[y * width + x] = sum;
}


// Column convolution
__global__ void gaussianBlurCol(
    const float* input, float* output,
    int width, int height,
    const float* kernel, int kSize, int kCenter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int k = 0; k < kSize; ++k) {
        int dy = y + (k - kCenter);
        if (dy < 0) dy = 0;
        if (dy >= height) dy = height - 1;
        sum += input[dy * width + x] * kernel[k];
    }
    output[y * width + x] = sum;
}


// Optimized row convolution kernel
__global__ void gaussianBlurRow_optimized(
    const float* input, float* output,
    int width, int height,
    const float* kernel, int kSize, int kCenter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    int base = y * width;
    for (int k = 0; k < kSize; ++k) {
        int dx = x + (k - kCenter);
        dx = dx < 0 ? 0 : (dx >= width ? width - 1 : dx);
        sum += input[base + dx] * kernel[k];
    }
    output[base + x] = sum;
}

// Optimized column convolution kernel 
__global__ void gaussianBlurCol_optimized(
    const float* input, float* output,
    int width, int height,
    const float* kernel, int kSize, int kCenter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    int base = y * width + x;
    for (int k = 0; k < kSize; ++k) {
        int dy = y + (k - kCenter);
        dy = dy < 0 ? 0 : (dy >= height ? height - 1 : dy);
        sum += input[dy * width + x] * kernel[k];
    }
    output[base] = sum;
}


//DoG Parallel Naive
// simple 1-D kernel: out[i] = curr[i] - prev[i]
__global__ void computeDoG(
    const float* __restrict__ curr,
    const float* __restrict__ prev,
          float* __restrict__ out,
    int             N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = curr[idx] - prev[idx];
    }
}


// Kernel that computes all DoG images for one octave in one pass:
//   for each idx in [0 .. N*(levels-1)):
//     scale = idx / N, pix = idx % N
//     out[idx] = flat[(scale+1)*N + pix] - flat[scale*N + pix]
__global__ void computeDoGAll(
    const float* __restrict__ flat,  // concatenated input images (levels × N)
          float* __restrict__ out,   // concatenated output images ((levels-1) × N)
    int           N,                // pixels per image
    int           levels            // total levels per octave
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (levels - 1);
    if (idx < total) {
        int scale = idx / N;
        int pix   = idx - scale * N;
        out[idx]  = flat[(scale+1)*N + pix] - flat[scale*N + pix];
    }
}


// one thread is mapped to one pixel (x,y)
__global__ void gradientKernel(
    const float* __restrict__ in,  // single‐channel input
    float*       __restrict__ out, // 2‐channel output (gx,gy interleaved)
    int width, int height
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) {
        // on border, zero‐pad
        if (x < width && y < height) {
            out[2*(y*width + x) + 0] = 0.f;
            out[2*(y*width + x) + 1] = 0.f;
        }
        return;
    }

    int idx = y*width + x;
    float l = in[y*width + (x-1)];
    float r = in[y*width + (x+1)];
    float u = in[(y-1)*width + x];
    float d = in[(y+1)*width + x];

    out[2*idx + 0] = 0.5f * (r - l);  // gx
    out[2*idx + 1] = 0.5f * (d - u);  // gy
}


// Kernel: compute gradients for all levels in one pass
//   in: flattened levels (levels × N), out: flattened gradients interleaved (levels × N × 2)
__global__ void gradientKernelAll(
    const float* __restrict__ in,    // input flattened: level*N + pix
          float* __restrict__ out,   // output flattened: 2*(level*N + pix)
    int           width,
    int           height,
    int           levels,
    int           N)               // pixels per image = width*height
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = levels * N;
    if (idx >= total) return;
    int lvl = idx / N;
    int pix = idx - lvl * N;
    int x = pix % width;
    int y = pix / width;

    float gx = 0.0f, gy = 0.0f;
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        int base = lvl * N;
        float l = in[base + (y*width + (x-1))];
        float r = in[base + (y*width + (x+1))];
        float u = in[base + ((y-1)*width + x)];
        float d = in[base + ((y+1)*width + x)];
        gx = 0.5f * (r - l);
        gy = 0.5f * (d - u);
    }
    int outIdx = 2 * idx;
    out[outIdx + 0] = gx;
    out[outIdx + 1] = gy;
}