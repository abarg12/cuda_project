#include "sift.hpp"

__global__ void identify_keypoints(float *image,
                                   unsigned int *keypoints,
                                   int image_size,
                                   int image_width,
                                   float contrast_thresh) 
{
    int image_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int image_down_idx = image_idx + image_size;
    int image_up_idx = image_idx + image_size * 2;

    if (image_idx < image_size) {
        if (std::abs(image[image_idx]) >= (0.8 * contrast_thresh)) {
            bool is_min = true, is_max = true;
            float val = image[image_idx];
            float neighbor = 0;
            int offset = 0;

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    offset = (dy * image_width) + dx;
                    if (image_idx + offset < 0 || image_idx + offset >= image_size) {
                        continue;
                    }

                    neighbor = image[image_down_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    neighbor = image[image_up_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    neighbor = image[image_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    if (!is_min && !is_max) {
                        return;
                    }
                }
            }
            
            keypoints[image_idx] = 1;
        }
    }

    return;
}


__global__ void generate_orientations_naive(float *devicePyramid,
                                            sift::Keypoint *deviceKeypoints,
                                            float *deviceDescriptors,
                                            int *deviceImgOffsets,
                                            int *deviceImgWidths,
                                            int *deviceImgHeights,
                                            int num_kps,
                                            int num_scales_per_octave)
{
    int kp_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (kp_idx < num_kps) {
        float pix_dist = sift::MIN_PIX_DIST * std::pow(2, deviceKeypoints[kp_idx].octave);
    }
}