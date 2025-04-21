#include "sift.hpp"


/* Summary:
 *     Performs initial keypoint selection by identifying scale-space extrema
 * Parameters:
 *   - image: 1D image data for 3 scales: current, below, and above.
 *   - keypoints: array of keypoint locations that will be populated with
 *                a '1' if there is a keypoint at that index in the 1D image
 *   - image_size: total number of pixels in a single image scale. The 'image'
 *                 parameter will contain 3*image_size number of pixels.
 *   - image_width: the width of a single image scale
 * Return:
 *     (void) popualtes the 'keypoints' with '1's for keypoint locations.
 *            The 'keypoints' vector is initialized as all '0's
 */
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


/* Summary:
 *     Helper function for the 'generate_orientations' kernel that applies
 *     a smoothing operation to the input histogram.
 * Parameters:
 *   - hist: the orientation histogram
 * Return:
 *     (void) modifies the values in the 'hist' array
 */
__device__ void smooth_histogram_device(float* hist) {
    float tmp[sift::N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < sift::N_BINS; j++) {
            int prev = (j - 1 + sift::N_BINS) % sift::N_BINS;
            int next = (j + 1) % sift::N_BINS;
            tmp[j] = (hist[prev] + hist[j] + hist[next]) / 3.0f;
        }
        for (int j = 0; j < sift::N_BINS; j++) {
            hist[j] = tmp[j];
        }
    }
}


/* Summary:
 *     Kernel that generates dominant orientations for a list of keypoints
 * Parameters:
 *   - devicePyramid: array of gradient data for the all images in scale space.
 *                    Concatenated [gx][gy] data for each image in a 1D array,
 *                    i.e. [img1_gx][img1_gy][img2_gx][img2_gy]... where img_gx
 *                    is a 1D array of all x-direction gradient data
 *   - deviceImgOffsets: 
 * Return:
 *     (void) modifies the values in the 'hist' array
 */
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
    float pix_dist = sift::MIN_PIX_DIST * powf(2.0f, kp.octave);
    float min_dist_from_border = fminf(fminf(kp.x, kp.y), 
                                       fminf(pix_dist * img_width - kp.x,
                                             pix_dist * img_height - kp.y));
    // discard keypoint if too close to image borders
    if (min_dist_from_border <= sqrtf(2.0f) * lambda_desc * kp.sigma) {
        return;
    }

    float hist[sift::N_BINS] = {0.0};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3.0f * patch_sigma;

    int x_start = roundf((kp.x - patch_radius) / pix_dist);
    int x_end   = roundf((kp.x + patch_radius) / pix_dist);
    int y_start = roundf((kp.y - patch_radius) / pix_dist);
    int y_end   = roundf((kp.y + patch_radius) / pix_dist);

    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            int clamped_x = max(0, min(x, img_width - 1));
            int clamped_y = max(0, min(y, img_height - 1));
            int idx = clamped_y * img_width + clamped_x;

            gx = devicePyramid[img_offset + idx];
            gy = devicePyramid[img_offset + idx + img_width * img_height];

            grad_norm = sqrtf(gx * gx + gy * gy);

            weight = expf(-(powf(x * pix_dist - kp.x, 2.0f) +
                              powf(y * pix_dist - kp.y, 2.0f)) /
                            (2.0f * patch_sigma * patch_sigma));

            theta = fmodf(atan2f(gy, gx) + 2.0f * sift::M_PIf, 2.0f * sift::M_PIf);
            bin = ((int)roundf(sift::N_BINS / (2.0f * sift::M_PIf) * theta)) % sift::N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram_device(hist);

    // extract reference orientations
    float ori_thresh = 0.8f;
    float ori_max = 0.0f;
    for (int i = 0; i < sift::N_BINS; i++) {
        if (hist[i] > ori_max) {
            ori_max = hist[i];
        }
    }

    for (int i = 0; i < sift::N_BINS; i++) {
        if (hist[i] >= ori_thresh * ori_max) {
            float prev = hist[(i - 1 + sift::N_BINS) % sift::N_BINS];
            float next = hist[(i + 1) % sift::N_BINS];
            if (prev > hist[i] || next > hist[i]) {
                continue;
            }

            float theta = 2.0f*sift::M_PIf*(i+1.0f)/sift::N_BINS + sift::M_PIf/sift::N_BINS*(prev-next)/(prev-2.0f*hist[i]+next);
            deviceOrientations[tid * sift::N_BINS + i] = theta;
        }
    }
}



__device__ void update_histograms_device(float* hist, float x, float y,
                                        float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= sift::N_HIST; i++) {
        x_i = (i - (1 + (float)sift::N_HIST) / 2.0f) * 2.0f * lambda_desc / sift::N_HIST;
        if (fabsf(x_i - x) > 2.0f * lambda_desc / sift::N_HIST)
            continue;
        for (int j = 1; j <= sift::N_HIST; j++) {
            y_j = (j - (1.0f + (float)sift::N_HIST) / 2.0f) * 2.0f * lambda_desc / sift::N_HIST;
            if (fabsf(y_j - y) > 2.0f * lambda_desc / sift::N_HIST)
                continue;

            float hist_weight = (1.0f - sift::N_HIST * 0.5f / lambda_desc * fabsf(x_i - x))
                              * (1.0f - sift::N_HIST * 0.5f / lambda_desc * fabsf(y_j - y));

            for (int k = 1; k <= sift::N_ORI; k++) {
                float theta_k = 2.0f * sift::M_PIf * (k - 1.0f) / sift::N_ORI;
                float theta_diff = fmodf(theta_k - theta_mn + 2.0f * sift::M_PIf, 2.0f * sift::M_PIf);
                if (fabsf(theta_diff) >= 2 * sift::M_PIf / sift::N_ORI)
                    continue;
                float bin_weight = 1.0f - sift::N_ORI * 0.5f / sift::M_PIf * fabsf(theta_diff);
                int hist_index = ((i - 1) * sift::N_HIST + (j - 1)) * sift::N_ORI + (k - 1);
                hist[hist_index] += hist_weight * bin_weight * contrib;
            }
        }
    }
}

__device__ void hists_to_vec_device(float* histograms, uint8_t* feature_vec)
{
    int size = sift::N_HIST * sift::N_HIST * sift::N_ORI;
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += histograms[i] * histograms[i];
    }
    norm = sqrtf(norm);
    float norm2 = 0.0f;
    for (int i = 0; i < size; i++) {
        histograms[i] = fminf(histograms[i], 0.2f * norm);
        norm2 += histograms[i] * histograms[i];
    }
    norm2 = sqrtf(norm2);
    for (int i = 0; i < size; i++) {
        float val = floorf(512.0f * histograms[i] / norm2);
        feature_vec[i] = static_cast<uint8_t>(min((int) val, 255));
    }
}

__global__ void generate_descriptors(float* devicePyramid,
                                           sift::Keypoint* deviceKeypoints,
                                           uint8_t* deviceKeypointDescriptors,
                                           float* thetas,
                                           int *deviceImgOffsets,
                                           int *deviceImgWidths,
                                           int *deviceImgHeights,
                                           int num_kps,
                                           int num_scales_per_octave,
                                           float lambda_desc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_kps)
        return;

    sift::Keypoint kp = deviceKeypoints[tid];
    float theta = thetas[tid];
     // Get the image index that the keypoint belongs to
    int img_idx = kp.octave * num_scales_per_octave + kp.scale;
    int img_offset = deviceImgOffsets[img_idx];
    int img_width = deviceImgWidths[img_idx];
    int img_height = deviceImgHeights[img_idx];
    float pix_dist = sift::MIN_PIX_DIST * powf(2.0f, kp.octave);
    float histograms[sift::N_HIST * sift::N_HIST * sift::N_ORI] = {0};

    float gx, gy, theta_mn, grad_norm, weight, contribution;

    float half_size = sqrtf(2.0f) * lambda_desc * kp.sigma * (sift::N_HIST + 1.0f) / sift::N_HIST;
    int x_start = roundf((kp.x - half_size) / pix_dist);
    int x_end = roundf((kp.x + half_size) / pix_dist);
    int y_start = roundf((kp.y - half_size) / pix_dist);
    int y_end = roundf((kp.y + half_size) / pix_dist);

    float cos_t = cosf(theta), sin_t = sinf(theta);
    float patch_sigma = lambda_desc * kp.sigma;

    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            int clamped_m = max(0, min(m, img_width - 1));
            int clamped_n = max(0, min(n, img_height - 1));
            int idx = clamped_n * img_width + clamped_m;

            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;


            // verify (x, y) is inside the description patch
            if (fmaxf(fabsf(x), fabsf(y)) > lambda_desc * (sift::N_HIST + 1.0f) / sift::N_HIST)
                continue;

            gx = devicePyramid[img_offset + idx];
            gy = devicePyramid[img_offset + idx + img_width * img_height];
            
            theta_mn = fmodf(atan2f(gy, gx) - theta + 4.0f * sift::M_PIf, 2.0f * sift::M_PIf);
            grad_norm = sqrtf(gx * gx + gy * gy);
            weight = expf(-(powf(m * pix_dist - kp.x, 2.0f) + powf(n * pix_dist - kp.y, 2.0f)) / (2.0f * patch_sigma * patch_sigma));
            contribution = weight * grad_norm;

            update_histograms_device(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    hists_to_vec_device(histograms, deviceKeypointDescriptors + tid * 128);
}

__global__ void generate_orientations_and_descriptors(float* devicePyramid,
                                                    sift::Keypoint* deviceKeypoints,
                                                    uint8_t* deviceKeypointDescriptors,
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
    float pix_dist = sift::MIN_PIX_DIST * std::pow(2, kp.octave);
    float min_dist_from_border = fminf(fminf(kp.x, kp.y), 
                                       fminf(pix_dist * img_width - kp.x,
                                             pix_dist * img_height - kp.y));
    // discard keypoint if too close to image borders
    if (min_dist_from_border <= sqrtf(2.0) * lambda_desc * kp.sigma) {
        return;
    }

    float hist[sift::N_BINS] = {0.0};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3.0 * patch_sigma;

    int x_start = roundf((kp.x - patch_radius) / pix_dist);
    int x_end   = roundf((kp.x + patch_radius) / pix_dist);
    int y_start = roundf((kp.y - patch_radius) / pix_dist);
    int y_end   = roundf((kp.y + patch_radius) / pix_dist);

    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            int clamped_x = max(0, min(x, img_width - 1));
            int clamped_y = max(0, min(y, img_height - 1));
            int idx = clamped_y * img_width + clamped_x;

            gx = devicePyramid[img_offset + idx];
            gy = devicePyramid[img_offset + idx + img_width * img_height];

            grad_norm = sqrtf(gx * gx + gy * gy);

            weight = expf(-(powf(x * pix_dist - kp.x, 2) +
                              powf(y * pix_dist - kp.y, 2)) /
                            (2 * patch_sigma * patch_sigma));

            theta = fmodf(atan2f(gy, gx) + 2 * sift::M_PIf, 2 * sift::M_PIf);
            bin = ((int)roundf(sift::N_BINS / (2 * sift::M_PIf) * theta)) % sift::N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram_device(hist);

    // extract reference orientations
    float ori_thresh = 0.8;
    float ori_max = 0.0;
    for (int i = 0; i < sift::N_BINS; i++) {
        if (hist[i] > ori_max) {
            ori_max = hist[i];
        }
    }

    float contribution, half_size, theta_mn;
    for (int i = 0; i < sift::N_BINS; i++) {
        if (hist[i] >= ori_thresh * ori_max) {
            float prev = hist[(i - 1 + sift::N_BINS) % sift::N_BINS];
            float next = hist[(i + 1) % sift::N_BINS];
            if (prev > hist[i] || next > hist[i]) {
                continue;
            }

            float theta = 2*sift::M_PIf*(i+1)/sift::N_BINS + sift::M_PIf/sift::N_BINS*(prev-next)/(prev-2*hist[i]+next);
            deviceOrientations[tid * sift::N_BINS + i] = theta;

            half_size = sqrtf(2) * lambda_desc * kp.sigma * (sift::N_HIST + 1.0) / sift::N_HIST;
            x_start = roundf((kp.x - half_size) / pix_dist);
            x_end = roundf((kp.x + half_size) / pix_dist);
            y_start = roundf((kp.y - half_size) / pix_dist);
            y_end = roundf((kp.y + half_size) / pix_dist);

            float cos_t = cosf(theta), sin_t = sinf(theta);
            float patch_sigma = lambda_desc * kp.sigma;
            float histograms[sift::N_HIST * sift::N_HIST * sift::N_ORI] = {0};

            for (int m = x_start; m <= x_end; m++) {
                for (int n = y_start; n <= y_end; n++) {
                    int clamped_m = max(0, min(m, img_width - 1));
                    int clamped_n = max(0, min(n, img_height - 1));
                    int idx = clamped_n * img_width + clamped_m;

                    // find normalized coords w.r.t. kp position and reference orientation
                    float x = ((m*pix_dist - kp.x)*cos_t
                            +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
                    float y = (-(m*pix_dist - kp.x)*sin_t
                            +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

                    // verify (x, y) is inside the description patch
                    if (fmaxf(fabsf(x), fabsf(y)) > lambda_desc * (sift::N_HIST + 1.0) / sift::N_HIST)
                        continue;

                    gx = devicePyramid[img_offset + idx];
                    gy = devicePyramid[img_offset + idx + img_width * img_height];
                    
                    theta_mn = fmodf(atan2f(gy, gx) - theta + 4 * sift::M_PIf, 2 * sift::M_PIf);
                    grad_norm = sqrtf(gx * gx + gy * gy);
                    weight = expf(-(powf(m * pix_dist - kp.x, 2) + powf(n * pix_dist - kp.y, 2)) / (2 * patch_sigma * patch_sigma));
                    contribution = weight * grad_norm;

                    update_histograms_device(histograms, x, y, contribution, theta_mn, lambda_desc);
                }
            }

            hists_to_vec_device(histograms, deviceKeypointDescriptors + 128 * (tid * sift::N_BINS + i));
        }
    }
}


// update_histograms_device: Receives the relative theta_mn
__device__ void update_histograms_device_atomic(float* histograms, float x_rotated, float y_rotated,
                                         float contrib, float theta_mn, float lambda_desc) // theta_mn is the RELATIVE orientation
{
    float x_i, y_j;
    for (int i = 1; i <= sift::N_HIST; i++) {
        x_i = (i - (1 + (float)sift::N_HIST) / 2.0f) * 2.0f * lambda_desc / sift::N_HIST;
        float d_x = fabsf(x_i - x_rotated);
        if (d_x > 2.0f * lambda_desc / sift::N_HIST) continue;

        for (int j = 1; j <= sift::N_HIST; j++) {
            y_j = (j - (1.0f + (float)sift::N_HIST) / 2.0f) * 2.0f * lambda_desc / sift::N_HIST;
            float d_y = fabsf(y_j - y_rotated);
            if (d_y > 2.0f * lambda_desc / sift::N_HIST) continue;

            float hist_weight_spatial = (1.0f - sift::N_HIST * 0.5f / lambda_desc * d_x)
                                      * (1.0f - sift::N_HIST * 0.5f / lambda_desc * d_y);

            for (int k = 1; k <= sift::N_ORI; k++) {
                float theta_k = 2.0f * sift::M_PIf * (k - 1.0f) / sift::N_ORI;
                 // Calculate difference using the relative theta_mn
                 float theta_diff = fmodf(theta_k - theta_mn + 2.0f * sift::M_PIf, 2.0f * sift::M_PIf);
                //  float d_theta = fabsf(theta_diff);

                //  if (d_theta > sift::M_PIf) d_theta = 2.0f * sift::M_PIf - d_theta;
                //  if (d_theta >= 2 * sift::M_PIf / sift::N_ORI) continue;
                if (fabsf(theta_diff) >= 2.0f * sift::M_PIf / sift::N_ORI)
                    continue;

                float bin_weight = 1.0f - sift::N_ORI * 0.5f / sift::M_PIf * fabsf(theta_diff);
                int hist_index = ((i - 1) * sift::N_HIST + (j - 1)) * sift::N_ORI + (k - 1);
                float total_contrib = hist_weight_spatial * bin_weight * contrib;

                atomicAdd((float*) &histograms[hist_index], total_contrib);
            }
        }
    }
}


__global__ void generate_descriptors_one_block_per_kp(
    float* devicePyramid,
    sift::Keypoint* deviceKeypoints, // List of keypoints to process
    uint8_t* deviceKeypointDescriptors, // Output buffer for descriptors
    float* thetas, // List of orientations corresponding to keypoints
    int *deviceImgOffsets,
    int *deviceImgWidths,
    int *deviceImgHeights,
    int num_kps,
    int num_scales_per_octave,
    float lambda_desc)
{
    int bIdx = blockIdx.x;

    if (bIdx >= num_kps)
        return;

    sift::Keypoint kp = deviceKeypoints[bIdx];
    float theta = thetas[bIdx];
    int img_idx = kp.octave * num_scales_per_octave + kp.scale;
    int img_offset = deviceImgOffsets[img_idx];
    int img_width = deviceImgWidths[img_idx];
    int img_height = deviceImgHeights[img_idx];
    float pix_dist = sift::MIN_PIX_DIST * pow(2.0f, kp.octave);

    __shared__ float histograms[sift::DESC_HIST_SHARED_SIZE];

    if (threadIdx.x == 0) {
        for(int i = 0; i < sift::DESC_HIST_SHARED_SIZE; ++i) {
           histograms[i] = 0.0f;
       }
    }
    __syncthreads();


    float cos_t = cosf(theta), sin_t = sinf(theta);
    float patch_sigma = lambda_desc * kp.sigma;

    float half_size = sqrtf(2.0f) * lambda_desc * kp.sigma * (sift::N_HIST + 1.0f) / sift::N_HIST;
    int x_start_patch = roundf((kp.x - half_size) / pix_dist);
    int x_end_patch = roundf((kp.x + half_size) / pix_dist);
    int y_start_patch = roundf((kp.y - half_size) / pix_dist);
    int y_end_patch = roundf((kp.y + half_size) / pix_dist);

    int patch_width = x_end_patch - x_start_patch + 1;
    int patch_height = y_end_patch - y_start_patch + 1;
    int total_neighbors = patch_width * patch_height;


    for (int neighbor_flat_idx = threadIdx.x; neighbor_flat_idx < total_neighbors; neighbor_flat_idx += blockDim.x)
    {
        int m = x_start_patch + (neighbor_flat_idx % patch_width);
        int n = y_start_patch + (neighbor_flat_idx / patch_width);

        int clamped_m = max(0, min(m, img_width - 1));
        int clamped_n = max(0, min(n, img_height - 1));
        int idx = clamped_n * img_width + clamped_m;

        float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
        float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

        if (fmaxf(fabsf(x), fabsf(y)) > lambda_desc * (sift::N_HIST + 1.0f) / sift::N_HIST) {
            continue;
        }

        float gx = devicePyramid[img_offset + idx];
        float gy = devicePyramid[img_offset + idx + img_width * img_height];

        float grad_norm = sqrtf(gx * gx + gy * gy);

        float weight = expf(-(powf(m * pix_dist - kp.x, 2.0f) + powf(n * pix_dist - kp.y, 2.0f)) / (2.0f * patch_sigma * patch_sigma));

        float contribution = weight * grad_norm;

        // Calculate the relative gradient orientation here, like in the original kernel
        float theta_mn = fmodf(atan2f(gy, gx) - theta + 4.0f * sift::M_PIf, 2.0f * sift::M_PIf);


        // Call the update function, passing the relative theta_mn
        update_histograms_device_atomic((float*) histograms, x, y, contribution, theta_mn, lambda_desc);

    }

    __syncthreads();

    if (threadIdx.x == 0) {
        // Call the finalization function
        hists_to_vec_device((float*) histograms, deviceKeypointDescriptors + bIdx * 128);
    }
}