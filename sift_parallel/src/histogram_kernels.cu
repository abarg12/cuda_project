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
                                            float *deviceOrientations,
                                            int *deviceImgOffsets,
                                            int *deviceImgWidths,
                                            int *deviceImgHeights,
                                            int num_kps,
                                            int num_scales_per_octave,
                                            float lambda_ori,
                                            float lambda_desc)
{
    // Get the current keypoint indexed by thread location in the grid
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // boundary checking, is the thread index a valid keypoint index?
    if (tid >= num_kps) {
        return;
    }

    sift::Keypoint kp = deviceKeypoints[tid];
    // Get the image index that the keypoint belongs to
    int img_idx = kp.octave * num_scales_per_octave + kp.scale;
    int img_offset = deviceImgOffsets[img_idx];
    int img_width = deviceImgWidths[img_idx];
    int img_height = deviceImgHeights[img_idx];
    // discard keypoint if too close to image borders
    float pix_dist = sift::MIN_PIX_DIST * std::pow(2, kp.octave);
    float min_dist_from_border = fminf(fminf(kp.x, kp.y), 
                                       fminf(pix_dist * img_width - kp.x,
                                             pix_dist * img_height - kp.y));
    if (min_dist_from_border <= sqrtf(2.0) * lambda_desc * kp.sigma) {
        return;
    }

    float hist[sift::N_BINS] = {0};
    int bin;
    float gx, gy, grad_norm, img_weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3.0f * patch_sigma;

    int x_start = roundf((kp.x - patch_radius) / pix_dist);
    int x_end   = roundf((kp.x + patch_radius) / pix_dist);
    int y_start = roundf((kp.y - patch_radius) / pix_dist);
    int y_end   = roundf((kp.y + patch_radius) / pix_dist);

    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            if (x < 0 || y < 0 || x >= img_width || y >= img_height) continue;

            gx = devicePyramid[img_offset + 2 * (y * img_width + x)];
            gy = devicePyramid[img_offset + 2 * (y * img_width + x) + 1];
            grad_norm = sqrtf(gx * gx + gy * gy);

            img_weight = __expf(-(powf(x * pix_dist - kp.x, 2.0f) +
                              powf(y * pix_dist - kp.y, 2.0f)) /
                            (2.0f * patch_sigma * patch_sigma));

            theta = fmodf(atan2f(gy, gx) + 2.0f * M_PI, 2.0f * M_PI);
            bin = ((int)roundf(sift::N_BINS / (2.0f * M_PI) * theta)) % sift::N_BINS;
            hist[bin] += img_weight * grad_norm;
        }
    }

    // For now: skip or do basic moving average
    float smooth[sift::N_BINS];
    for (int i = 0; i < sift::N_BINS; i++) {
        float prev = hist[(i - 1 + sift::N_BINS) % sift::N_BINS];
        float next = hist[(i + 1) % sift::N_BINS];
        smooth[i] = 0.25f * prev + 0.5f * hist[i] + 0.25f * next;
    }
    for (int i = 0; i < sift::N_BINS; i++) hist[i] = smooth[i];

    // extract reference orientations
    float ori_thresh = 0.8f;
    float ori_max = 0.0f;
    for (int j = 0; j < sift::N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }

    for (int j = 0; j < sift::N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j - 1 + sift::N_BINS) % sift::N_BINS];
            float next = hist[(j + 1) % sift::N_BINS];
            // Use -1.0 to indicate no bin value added at this location
            if (prev > hist[j] || next > hist[j]) {
                // deviceOrientations[tid * sift::N_BINS + j] = -1.0f;
                continue;
            };

            // Quadratic interpolation for peak
            // float interp = 0.5f * (prev - next) / (prev - 2.0f * hist[j] + next);
            // float theta = 2.0f * M_PI * (j + interp + 0.5f) / sift::N_BINS;
            float theta = 2*M_PI*(j+1)/sift::N_BINS + M_PI/sift::N_BINS*(prev-next)/(prev-2*hist[j]+next);
            deviceOrientations[tid * sift::N_BINS + j] = theta;
        }
    }
}