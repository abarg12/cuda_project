#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <cuda_runtime.h>
#include <chrono>

#include "sift.hpp"
#include "image.hpp"
#include "histogram_kernels.cu"


#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}


namespace sift {

ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
        }
        // prepare base image for next octave
        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
                                        Interpolation::NEAREST);
    }
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            Image diff = img_pyramid.octaves[i][j];
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= img_pyramid.octaves[i][j-1].data[pix_idx];
            }
            dog_pyramid.octaves[i].push_back(diff);
        }
    }
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return std::make_tuple(offset_s, offset_x, offset_y);
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        float offset_s, offset_x, offset_y;
        std::tie(offset_s, offset_x, offset_y) = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image& img = octave[j];
            for (int x = 1; x < img.width-1; x++) {
                for (int y = 1; y < img.height-1; y++) {
                    if (std::abs(img.get_pixel(x, y, 0)) < 0.8*contrast_thresh) {
                        continue;
                    }
                    if (point_is_extremum(octave, j, x, y)) {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
                                                                      edge_thresh);
                        if (kp_is_valid) {
                            keypoints.push_back(kp);
                        }
                    }
                }
            }
        }
    }
    return keypoints;
}

std::vector<Keypoint> find_keypoints_parallel_naive(const ScaleSpacePyramid& dog_pyramid,
                                                    float contrast_thresh,
                                                    float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image &img = octave[j];
            const Image &img_down = octave[j - 1];
            const Image &img_up = octave[j + 1];


            // Allocate device memory for the image at 3 different scales
            // the 3 scales get compared for identifying keypoints
            int img_size_b = img.size * sizeof(float); 
            float *deviceImage;
            CUDA_CHECK(
                cudaMalloc((void **) &deviceImage, img_size_b * 3)
            );
            // Copy over host images to device memory
            CUDA_CHECK(
                cudaMemcpy(deviceImage,
                           img.data,
                           img_size_b,
                           cudaMemcpyHostToDevice)
            );
            CUDA_CHECK(
                cudaMemcpy(deviceImage + img.size,
                           img_down.data,
                           img_size_b,
                           cudaMemcpyHostToDevice)
            );
            CUDA_CHECK(
                cudaMemcpy(deviceImage + 2 * img.size,
                           img_up.data,
                           img_size_b,
                           cudaMemcpyHostToDevice)
            );

            // Allocate device memory for output buffer which is an array with
            // the same size as the image where a 0 indicates a non-keypoint at
            // the corresponding pixel position and a 1 indicates a keypoint
            unsigned int *deviceKeypointOutput;
            unsigned int *hostKeypointOutput = new unsigned int[img.size];
            
            CUDA_CHECK(
                cudaMalloc((void **) &deviceKeypointOutput, img.size * sizeof(int))
            );
            // reset keypoint indicator buffer to all 0s
            CUDA_CHECK(
                cudaMemset(deviceKeypointOutput, 0, img.size * sizeof(int))
            );

            dim3 blockDim(512);
            dim3 gridDim((img.size + blockDim.x - 1) / blockDim.x);
            identify_keypoints<<<gridDim, blockDim>>>(deviceImage,
                                                      deviceKeypointOutput,
                                                      img.size,
                                                      img.width,
                                                      contrast_thresh);

            CUDA_CHECK(
                cudaMemcpy(hostKeypointOutput,
                           deviceKeypointOutput,
                           img.size * sizeof(int),
                           cudaMemcpyDeviceToHost)
            );

            int totalkp = 0;
            for (int ind = 0; ind < img.size; ind++) {
                if (hostKeypointOutput[ind]) {
                    totalkp++;
                }
            }

            for (int x = 1; x < img.width-1; x++) {
                for (int y = 1; y < img.height-1; y++) {
                    int img_idx = x + y * img.width;
                    if (hostKeypointOutput[img_idx]) {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
                                                                      edge_thresh);
                        if (kp_is_valid) {
                            keypoints.push_back(kp);
                        }
                    }
                }
            }

            CUDA_CHECK(cudaFree(deviceImage));
            CUDA_CHECK(cudaFree(deviceKeypointOutput));
            delete[] hostKeypointOutput;
        }
    }
    return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].reserve(grad_pyramid.imgs_per_octave);
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            Image grad(width, height, 2);
            float gx, gy;
            for (int x = 1; x < grad.width-1; x++) {
                for (int y = 1; y < grad.height-1; y++) {
                    gx = (pyramid.octaves[i][j].get_pixel(x+1, y, 0)
                         -pyramid.octaves[i][j].get_pixel(x-1, y, 0)) * 0.5;
                    grad.set_pixel(x, y, 0, gx);
                    gy = (pyramid.octaves[i][j].get_pixel(x, y+1, 0)
                         -pyramid.octaves[i][j].get_pixel(x, y-1, 0)) * 0.5;
                    grad.set_pixel(x, y, 1, gy);
                }
            }
            grad_pyramid.octaves[i].push_back(grad);
        }
    }
    return grad_pyramid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                              /(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

std::vector<std::vector<float>> find_keypoint_orientations_parallel_naive(std::vector<Keypoint>& kps,
                                                                          const ScaleSpacePyramid& grad_pyramid,
                                                                          float lambda_ori,
                                                                          float lambda_desc)
{
    int pyramidOneDimSize = 0;
    for (int i = 0; i < grad_pyramid.num_octaves; i++) {
        for (int j = 0; j < grad_pyramid.imgs_per_octave; j++) {
            pyramidOneDimSize += grad_pyramid.octaves[i][j].size;
        }
    }

    // Allocate global memory to hold the input gradient pyramid
    float *devicePyramid;
    CUDA_CHECK(
        cudaMalloc((void**) &devicePyramid, pyramidOneDimSize * sizeof(float)));

    // Copy gradient pyramid data into global memory
    int image_idx = 0;
    for (int i = 0; i < grad_pyramid.num_octaves; i++) {
        for (int j = 0; j < grad_pyramid.imgs_per_octave; j++) {
            CUDA_CHECK(
                cudaMemcpy(devicePyramid + image_idx,
                           grad_pyramid.octaves[i][j].data,
                           grad_pyramid.octaves[i][j].size * sizeof(float),
                           cudaMemcpyHostToDevice));
            image_idx += grad_pyramid.octaves[i][j].size;
        }
    }

    // Allocate global memory and host memory to hold keypoints
    Keypoint *deviceKeypoints;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypoints, kps.size() * sizeof(Keypoint)));
    
    Keypoint *hostKeypoints = kps.data();
    
    // Copy keypoints from host to device memory
    CUDA_CHECK(
        cudaMemcpy(deviceKeypoints, hostKeypoints, 
                   kps.size() * sizeof(Keypoint), cudaMemcpyHostToDevice));


    // Allocate global memory and host memory to hold keypoint orientations
    float *hostOrientations = new float[kps.size() * N_BINS];
    float *deviceOrientations;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceOrientations, kps.size() * N_BINS * sizeof(float)));
    CUDA_CHECK(
        cudaMemset(deviceOrientations, 0, kps.size() * N_BINS * sizeof(float)));
    

    // Compute image dimensions in order to index into the gradient pyramid
    int total_images = grad_pyramid.num_octaves * grad_pyramid.imgs_per_octave;
    std::vector<int> imageOffsets(total_images);
    std::vector<int> imageWidths(total_images);
    std::vector<int> imageHeights(total_images);
    int *deviceImgOffsets, *deviceImgWidths, *deviceImgHeights;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceImgOffsets, total_images * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc((void **) &deviceImgWidths, total_images * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc((void **) &deviceImgHeights, total_images * sizeof(int)));


    int offset = 0;
    int idx = 0;
    for (int i = 0; i < grad_pyramid.num_octaves; i++) {
        for (int j = 0; j < grad_pyramid.imgs_per_octave; j++) {
            const Image& img = grad_pyramid.octaves[i][j];
            imageOffsets[idx] = offset;
            imageWidths[idx] = img.width;
            imageHeights[idx] = img.height;
            offset += img.size;
            idx++;
        }
    }

    CUDA_CHECK(
        cudaMemcpy(deviceImgOffsets, imageOffsets.data(), total_images * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(deviceImgWidths, imageWidths.data(), total_images * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(deviceImgHeights, imageHeights.data(), total_images * sizeof(int), cudaMemcpyHostToDevice));


    dim3 blockDim(512);
    dim3 gridDim((kps.size() + blockDim.x - 1) / blockDim.x);
    generate_orientations_naive<<<gridDim, blockDim>>>(devicePyramid,
                                                       deviceKeypoints,
                                                       deviceOrientations,
                                                       deviceImgOffsets,
                                                       deviceImgWidths,
                                                       deviceImgHeights,
                                                       kps.size(),
                                                       grad_pyramid.imgs_per_octave,
                                                       lambda_ori,
                                                       lambda_desc);

    CUDA_CHECK(
        cudaMemcpy(hostOrientations, deviceOrientations, kps.size() * N_BINS * sizeof(float), cudaMemcpyDeviceToHost));

    // create output std::vector of keypoint orientations
    std::vector<std::vector<float>> orientation_outvec;
    for (int i = 0; i < kps.size(); i++) {
        orientation_outvec.push_back({});
        for (int j = 0; j < N_BINS; j++) {
            if (hostOrientations[i * N_BINS + j] != 0.0 && !isnan(hostOrientations[i * N_BINS + j])) {
                orientation_outvec[i].push_back(hostOrientations[i * N_BINS + j]);
            }
        }
    }

    CUDA_CHECK(cudaFree(devicePyramid));
    CUDA_CHECK(cudaFree(deviceKeypoints));
    CUDA_CHECK(cudaFree(deviceOrientations));
    CUDA_CHECK(cudaFree(deviceImgOffsets));
    CUDA_CHECK(cudaFree(deviceImgWidths));
    CUDA_CHECK(cudaFree(deviceImgHeights));
    delete[] hostOrientations;

    return orientation_outvec;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++) {
        float val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    //accumulate samples into histograms
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}


void compute_keypoint_descriptors_parallel_naive(std::vector<Keypoint>& kps,
                                                std::vector<float> thetas,
                                                const ScaleSpacePyramid& grad_pyramid,
                                                float lambda_desc)
{
    int pyramidOneDimSize = 0;
    for (int i = 0; i < grad_pyramid.num_octaves; i++) {
        for (int j = 0; j < grad_pyramid.imgs_per_octave; j++) {
            pyramidOneDimSize += grad_pyramid.octaves[i][j].size;
        }
    }

    // Allocate global memory to hold the input gradient pyramid
    float *devicePyramid;
    CUDA_CHECK(
        cudaMalloc((void**) &devicePyramid, pyramidOneDimSize * sizeof(float)));

    // Copy gradient pyramid data into global memory
    int image_idx = 0;
    for (int i = 0; i < grad_pyramid.num_octaves; i++) {
        for (int j = 0; j < grad_pyramid.imgs_per_octave; j++) {
            CUDA_CHECK(
                cudaMemcpy(devicePyramid + image_idx,
                           grad_pyramid.octaves[i][j].data,
                           grad_pyramid.octaves[i][j].size * sizeof(float),
                           cudaMemcpyHostToDevice));
            image_idx += grad_pyramid.octaves[i][j].size;
        }
    }

    
    // Allocate global memory and host memory to hold keypoints
    Keypoint *deviceKeypoints;
    uint8_t *deviceKeypointDescriptors;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypoints, kps.size() * sizeof(Keypoint)));
    Keypoint *hostKeypoints = kps.data();
    // Copy keypoints from host to device memory
    CUDA_CHECK(
        cudaMemcpy(deviceKeypoints, hostKeypoints, 
                   kps.size() * sizeof(Keypoint), cudaMemcpyHostToDevice));

    // Allocate device memory for each keypoint's descriptor
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypointDescriptors, kps.size() * 128 * sizeof(uint8_t)));
    // Zero out device memory
    CUDA_CHECK(
        cudaMemset(deviceKeypointDescriptors, 0, kps.size() * 128 * sizeof(uint8_t)));


    // Allocate global memory for thetas
    float *deviceThetas;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceThetas, thetas.size() * sizeof(float)));
    // Copy host thetas to global memory
    CUDA_CHECK(
        cudaMemcpy(deviceThetas, thetas.data(),
                   thetas.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Compute image dimensions in order to index into the gradient pyramid
    int total_images = grad_pyramid.num_octaves * grad_pyramid.imgs_per_octave;
    std::vector<int> imageOffsets(total_images);
    std::vector<int> imageWidths(total_images);
    std::vector<int> imageHeights(total_images);
    int *deviceImgOffsets, *deviceImgWidths, *deviceImgHeights;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceImgOffsets, total_images * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc((void **) &deviceImgWidths, total_images * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc((void **) &deviceImgHeights, total_images * sizeof(int)));

    int offset = 0;
    int idx = 0;
    for (int i = 0; i < grad_pyramid.num_octaves; i++) {
        for (int j = 0; j < grad_pyramid.imgs_per_octave; j++) {
            const Image& img = grad_pyramid.octaves[i][j];
            imageOffsets[idx] = offset;
            imageWidths[idx] = img.width;
            imageHeights[idx] = img.height;
            offset += img.size;
            idx++;
        }
    }

    CUDA_CHECK(
        cudaMemcpy(deviceImgOffsets, imageOffsets.data(), total_images * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(deviceImgWidths, imageWidths.data(), total_images * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(deviceImgHeights, imageHeights.data(), total_images * sizeof(int), cudaMemcpyHostToDevice));


    dim3 blockDim(64);
    dim3 gridDim((kps.size() + blockDim.x - 1) / blockDim.x);
    generate_descriptors_naive<<<gridDim, blockDim>>>(devicePyramid,
                                                       deviceKeypoints,
                                                       deviceKeypointDescriptors,
                                                       deviceThetas,
                                                       deviceImgOffsets,
                                                       deviceImgWidths,
                                                       deviceImgHeights,
                                                       kps.size(),
                                                       grad_pyramid.imgs_per_octave,
                                                       lambda_desc);

    // Copy the descriptors back from the device
    for (size_t i = 0; i < kps.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(kps[i].descriptor.data(), deviceKeypointDescriptors + i * 128,
                             128 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }

    // // build feature vector (descriptor) from histograms
    // hists_to_vec(histograms, kp.descriptor);
    CUDA_CHECK(cudaFree(devicePyramid));
    CUDA_CHECK(cudaFree(deviceKeypoints));
    CUDA_CHECK(cudaFree(deviceKeypointDescriptors));
    CUDA_CHECK(cudaFree(deviceThetas));
    CUDA_CHECK(cudaFree(deviceImgOffsets));
    CUDA_CHECK(cudaFree(deviceImgWidths));
    CUDA_CHECK(cudaFree(deviceImgHeights));
}



/******************************************************************************
* The serial 'main' function which calls all necessary functions to compute the
* keypoints and descriptors.
******************************************************************************/
std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);

    printf("--------------------------------------------------\n");
    printf("RUNNING SERIAL SIFT\n");

    // Generate the Gaussian Pyramid
    auto t_start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("[Serial] generate_gaussian_pyramid elapsed time: %f ms\n", elapsed_ms);

    // Generate the DoG Pyramid
    t_start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("[Serial] generate_dog_pyramid elapsed time: %f ms\n", elapsed_ms);

    // Find keypoints
    t_start = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("[Serial] find_keypoints elapsed time: %f ms\n", elapsed_ms);
    
    // Generate gradient pyramid
    t_start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("[Serial] generate_gradient_pyramid elapsed time: %f ms\n", elapsed_ms);

    // Generate keypoint descriptors
    t_start = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> kps;
    for (Keypoint& kp_tmp : tmp_kps) {
        std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                     lambda_ori, lambda_desc);
        
        for (float theta : orientations) {
            Keypoint kp = kp_tmp;
            compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
            kps.push_back(kp);
        }
    }
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("[Serial] keypoint descriptor generation elapsed time: %f ms\n", elapsed_ms);

    return kps;
}

/******************************************************************************
* The parallel 'main' function which calls all necessary functions to compute
* the keypoints and descriptors.
******************************************************************************/
std::vector<Keypoint> find_keypoints_and_descriptors_parallel_naive(
    const Image& img,
    float sigma_min,
    int num_octaves,
    int scales_per_octave, 
    float contrast_thresh,
    float edge_thresh, 
    float lambda_ori,
    float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    // The timer objects used to record the overall parallel runtime for each
    // phase of SIFT. These time values include allocating and moving memory
    // between the host and device and other overhead like error checking
    cudaEvent_t startEvent, stopEvent;
    float elapsedTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    printf("--------------------------------------------------\n");
    printf("RUNNING PARALLEL SIFT (NAIVE)\n");

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);

    // Generate the Gaussian Pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] generate_gaussian_pyramid elapsed time %f ms\n", elapsedTime);

    // Generate the DoG Pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] generate_dog_pyramid elapsed time %f ms\n", elapsedTime);

    // Find keypoints
    cudaEventRecord(startEvent, 0);
    std::vector<Keypoint> tmp_kps = find_keypoints_parallel_naive(dog_pyramid, contrast_thresh, edge_thresh);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] find_keypoints_parallel elapsed time %f ms\n", elapsedTime);

    // Generate gradient pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] generate_gradient_pyramid elapsed time %f ms\n", elapsedTime);

    // Generate keypoint descriptors
    cudaEventRecord(startEvent, 0);
    std::vector<std::vector<float>> orientations_parallel = find_keypoint_orientations_parallel_naive
                                                                (tmp_kps, grad_pyramid, lambda_ori, lambda_desc);

    std::vector<Keypoint> kps;
    for (int i = 0; i < orientations_parallel.size(); i++) {
        for (float theta : orientations_parallel[i]) {
            Keypoint kp = tmp_kps[i];
            kps.push_back(kp);
        }
    }
    std::vector<float> one_dim_orienations;
    for (const auto& row : orientations_parallel) {
        one_dim_orienations.insert(one_dim_orienations.end(), row.begin(), row.end());
    }
    compute_keypoint_descriptors_parallel_naive(kps, one_dim_orienations, grad_pyramid, lambda_desc);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] keypoint descriptor generation elapsed time %f ms\n", elapsedTime);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return kps;
}


/******************************************************************************
* The parallel 'main' function which calls all necessary functions to compute
* the keypoints and descriptors.
******************************************************************************/
std::vector<Keypoint> find_keypoints_and_descriptors_parallel(
    const Image& img,
    float sigma_min,
    int num_octaves,
    int scales_per_octave, 
    float contrast_thresh,
    float edge_thresh, 
    float lambda_ori,
    float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    // The timer objects used to record the overall parallel runtime for each
    // phase of SIFT. These time values include allocating and moving memory
    // between the host and device and other overhead like error checking
    cudaEvent_t startEvent, stopEvent;
    float elapsedTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    printf("--------------------------------------------------\n");
    printf("RUNNING PARALLEL SIFT (OPTIMIZED)\n");

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);

    // Generate the Gaussian Pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] generate_gaussian_pyramid elapsed time %f ms\n", elapsedTime);

    // Generate the DoG Pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] generate_dog_pyramid elapsed time %f ms\n", elapsedTime);

    // Find keypoints
    cudaEventRecord(startEvent, 0);
    std::vector<Keypoint> tmp_kps = find_keypoints_parallel_naive(dog_pyramid, contrast_thresh, edge_thresh);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] find_keypoints_parallel elapsed time %f ms\n", elapsedTime);

    // Generate gradient pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] generate_gradient_pyramid elapsed time %f ms\n", elapsedTime);

    // Generate keypoint descriptors
    cudaEventRecord(startEvent, 0);
    std::vector<Keypoint> kps;  
    for (Keypoint& kp_tmp : tmp_kps) {
        std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                     lambda_ori, lambda_desc);
        for (float theta : orientations) {
            Keypoint kp = kp_tmp;
            compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
            kps.push_back(kp);
        }
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("[Parallel] keypoint descriptor generation elapsed time %f ms\n", elapsedTime);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return kps;
}


float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<Keypoint>& a,
                                                       std::vector<Keypoint>& b,
                                                       float thresh_relative,
                                                       float thresh_absolute)
{
    assert(a.size() >= 2 && b.size() >= 2);

    std::vector<std::pair<int, int>> matches;

    for (int i = 0; i < a.size(); i++) {
        // find two nearest neighbours in b for current keypoint from a
        int nn1_idx = -1;
        float nn1_dist = 100000000, nn2_dist = 100000000;
        for (int j = 0; j < b.size(); j++) {
            float dist = euclidean_dist(a[i].descriptor, b[j].descriptor);
            if (dist < nn1_dist) {
                nn2_dist = nn1_dist;
                nn1_dist = dist;
                nn1_idx = j;
            } else if (nn1_dist <= dist && dist < nn2_dist) {
                nn2_dist = dist;
            }
        }
        if (nn1_dist < thresh_relative*nn2_dist && nn1_dist < thresh_absolute) {
            matches.push_back({i, nn1_idx});
        }
    }
    return matches;
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}

Image draw_matches(const Image& a, const Image& b, std::vector<Keypoint>& kps_a,
                   std::vector<Keypoint>& kps_b, std::vector<std::pair<int, int>> matches)
{
    Image res(a.width+b.width, std::max(a.height, b.height), 3);

    for (int i = 0; i < a.width; i++) {
        for (int j = 0; j < a.height; j++) {
            res.set_pixel(i, j, 0, a.get_pixel(i, j, 0));
            res.set_pixel(i, j, 1, a.get_pixel(i, j, a.channels == 3 ? 1 : 0));
            res.set_pixel(i, j, 2, a.get_pixel(i, j, a.channels == 3 ? 2 : 0));
        }
    }
    for (int i = 0; i < b.width; i++) {
        for (int j = 0; j < b.height; j++) {
            res.set_pixel(a.width+i, j, 0, b.get_pixel(i, j, 0));
            res.set_pixel(a.width+i, j, 1, b.get_pixel(i, j, b.channels == 3 ? 1 : 0));
            res.set_pixel(a.width+i, j, 2, b.get_pixel(i, j, b.channels == 3 ? 2 : 0));
        }
    }

    for (auto& m : matches) {
        Keypoint& kp_a = kps_a[m.first];
        Keypoint& kp_b = kps_b[m.second];
        draw_line(res, kp_a.x, kp_a.y, a.width+kp_b.x, kp_b.y);
    }
    return res;
}

} // namespace sift
