// CUDA kernels for separable Gaussian blur
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cassert>


#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err);                                             \
    }                                                          \
} while (0)

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

// GPU-based gaussian_blur replacement
Image gaussian_blur_gpu(const Image& img, float sigma)
{
    assert(img.channels == 1);
    int width = img.width;
    int height = img.height;

    // Build 1D Gaussian kernel on host
    int kSize = static_cast<int>(ceilf(6 * sigma));
    if (kSize % 2 == 0) ++kSize;
    int kCenter = kSize / 2;
    std::vector<float> h_kernel(kSize);
    float sum = 0;
    for (int i = 0; i < kSize; ++i) {
        int x = i - kCenter;
        float v = expf(-(x*x) / (2 * sigma * sigma));
        h_kernel[i] = v;
        sum += v;
    }
    for (auto &v : h_kernel) v /= sum;

    // Allocate device memory
    float *d_in, *d_tmp, *d_out, *d_kernel;
    size_t imgBytes = width * height * sizeof(float);
    size_t kernelBytes = kSize * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_in,    imgBytes));
    CHECK_CUDA(cudaMalloc(&d_tmp,   imgBytes));
    CHECK_CUDA(cudaMalloc(&d_out,   imgBytes));
    CHECK_CUDA(cudaMalloc(&d_kernel,kernelBytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in,     img.data, imgBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), kernelBytes, cudaMemcpyHostToDevice));

    // Launch parameters
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1)/block.x,
              (height+ block.y - 1)/block.y);

    // Row blur
    gaussianBlurRow<<<grid, block>>>(d_in, d_tmp, width, height, d_kernel, kSize, kCenter);
    CHECK_CUDA(cudaGetLastError());
    // Column blur
    gaussianBlurCol<<<grid, block>>>(d_tmp, d_out, width, height, d_kernel, kSize, kCenter);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back
    Image filtered(width, height, 1);
    CHECK_CUDA(cudaMemcpy(filtered.data, d_out, imgBytes, cudaMemcpyDeviceToHost));

    // Free
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    cudaFree(d_kernel);

    return filtered;
}

// C-callable parallel pyramid generator
extern "C"
ScaleSpacePyramid generate_gaussian_pyramid_parallel(
    const Image& img, float sigma_min,
    int num_octaves, int scales_per_octave)
{
    // same setup as serial
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur_gpu(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals(imgs_per_octave);
    sigma_vals[0] = base_sigma;
    for (int i = 1; i < imgs_per_octave; i++) {
        float prev = base_sigma * std::pow(k, i-1);
        float total = k * prev;
        sigma_vals[i] = std::sqrt(total*total - prev*prev);
    }

    ScaleSpacePyramid pyramid = { num_octaves, imgs_per_octave, std::vector<std::vector<Image>>(num_octaves) };
    for (int o = 0; o < num_octaves; ++o) {
        auto& octave = pyramid.octaves[o];
        octave.reserve(imgs_per_octave);
        octave.push_back(std::move(base_img));
        for (int s = 1; s < imgs_per_octave; ++s) {
            const Image& prev = octave.back();
            octave.push_back(gaussian_blur_gpu(prev, sigma_vals[s]));
        }
        const Image& next_base = octave[imgs_per_octave - 3];
        base_img = next_base.resize(next_base.width/2, next_base.height/2, Interpolation::NEAREST);
    }
    return pyramid;
}

