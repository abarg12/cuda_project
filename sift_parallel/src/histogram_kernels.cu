#include "sift.hpp"


/* Summary:
 *     Performs initial keypoint selection by identifying scale-space extrema
 * Parameters:
 *   - image: input array with image data for 3 scales
 *   - keypoints: output array of keypoint locations that is populated with
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
    // pixel index within the image is assigned to a single thread
    int image_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // the same thread also accesses the image one scale down
    int image_down_idx = image_idx + image_size;
    // the same thread also accesses the image one scale up
    int image_up_idx = image_idx + image_size * 2;

    // boundary checking to make sure thread index is within image size
    if (image_idx < image_size) {
        // checks if pixel location has high enough contrast to be a keypoint
        if (std::abs(image[image_idx]) >= (0.8 * contrast_thresh)) {
            bool is_min = true, is_max = true;
            float val = image[image_idx];
            float neighbor = 0;
            int offset = 0;

            // compares the keypoint candidate to the neighbors above and below it
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    offset = (dy * image_width) + dx;
                    if (image_idx + offset < 0 || image_idx + offset >= image_size) {
                        continue;
                    }

                    // keypoint comparison to neighbor pixels at same scale
                    neighbor = image[image_down_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    // keypoint comparison to neighbor pixels one scale down
                    neighbor = image[image_up_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    // keypoint comparison to neighbor pixels one scale up
                    neighbor = image[image_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    // keypoint candidate is only valid keypoint if extrema
                    if (!is_min && !is_max) {
                        return;
                    }
                }
            }
            
            // fill the global output array with an indication that the pixel
            // at location 'image_idx' is a keypoint
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
 *     (void) modifies the values in the 'hist' array in place
 */
__device__ void smooth_histogram_device(float* hist) {
    float tmp[sift::N_BINS];
    // average histogram values with neighbors 6 times
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < sift::N_BINS; j++) {
            int prev = (j - 1 + sift::N_BINS) % sift::N_BINS;
            int next = (j + 1) % sift::N_BINS;
            tmp[j] = (hist[prev] + hist[j] + hist[next]) / 3.0f;
        }

        // assign results to the 'hist' array, modification in-place
        for (int j = 0; j < sift::N_BINS; j++) {
            hist[j] = tmp[j];
        }
    }
}


/* Summary:
 *     Kernel that generates dominant orientations for a list of keypoints
 * Parameters:
 *   - gradPyramid: input array of gradient data for all images in scale space.
 *                  Concatenated [gx][gy] data for each image in a 1D array,
 *                  i.e. [img1_gx][img1_gy][img2_gx][img2_gy]... where img_gx
 *                  is a 1D array of all x-direction gradient data
 *   - keypoints: input array of all keypoints to process
 *   - orientations: output array which will hold the output orientations
 *                   produced for each keypoint. The indices of this array
 *                   match the keypoint indices in 'keypoints'
 *   - imgOffsets: input array holding the start index locations for all
 *                 images in the scale space 'gradPyramid'. This was necessary
 *                 bookkeeping for accessing the gradPyramid as a combined
 *                 1D array of images at all octaves and scales.
 *                 i.e. {img1_offset, img2_offset, img3_offset, ...}
 *   - imgWidths: input array holding the image widths for all images in the
 *                scale space array 'gradPyramid' 
 *                i.e. {img1_width, img2_width, img3_width, ...}
 *   - imgHeights: input array holding the image heights for all images in the
 *                 scale space array 'gradPyramid'
 *                 i.e. {img1_height, img2_height, img3_height, ...}
 *   - num_kps: number of keypoints to be processed
 *   - num_scales_per_octave: number of scales for each octave in 'gradPyramid'
 *   - lambda_ori: controls size of orientation neighborhood for a keypoint
 *   - lambda_desc: thresholds the keypoints for proximity to border
 * Return:
 *     (void) modifies the orientations by reference
 * Notes:
 *     Granularity is one thread per keypoint. Each thread will process all
 *     orientations for the keypoint and determine which are dominant.
 */
__global__ void generate_orientations(float *gradPyramid,
                                      sift::Keypoint *keypoints,
                                      float *orientations,
                                      int *imgOffsets,
                                      int *imgWidths,
                                      int *imgHeights,
                                      int num_kps,
                                      int num_scales_per_octave,
                                      float lambda_ori,
                                      float lambda_desc)
{
    // Get the current keypoint indexed by thread location in the grid
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // boundary checking, confirms that thread index is a valid keypoint index
    if (tid >= num_kps) {
        return;
    }

    sift::Keypoint kp = keypoints[tid];
    // Get the image index that the keypoint belongs to
    int img_idx = kp.octave * num_scales_per_octave + kp.scale;
    // bookkeeping that lets us index into the gradPyramid array
    int img_offset = imgOffsets[img_idx];
    int img_width = imgWidths[img_idx];
    int img_height = imgHeights[img_idx];

    float pix_dist = sift::MIN_PIX_DIST * powf(2.0f, kp.octave);
    float min_dist_from_border = fminf(fminf(kp.x, kp.y), 
                                       fminf(pix_dist * img_width - kp.x,
                                             pix_dist * img_height - kp.y));
    // discard keypoint if too close to image borders
    if (min_dist_from_border <= sqrtf(2.0f) * lambda_desc * kp.sigma) {
        return;
    }

    // local histogram that stores keypoint orientations
    float hist[sift::N_BINS] = {0.0};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3.0f * patch_sigma;

    // variables that determine the pixel neighborhood around a keypoint
    int x_start = roundf((kp.x - patch_radius) / pix_dist);
    int x_end   = roundf((kp.x + patch_radius) / pix_dist);
    int y_start = roundf((kp.y - patch_radius) / pix_dist);
    int y_end   = roundf((kp.y + patch_radius) / pix_dist);

    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            int clamped_x = max(0, min(x, img_width - 1));
            int clamped_y = max(0, min(y, img_height - 1));
            int idx = clamped_y * img_width + clamped_x;

            // get the X and Y direction gradients from the pyramid
            gx = gradPyramid[img_offset + idx];
            gy = gradPyramid[img_offset + idx + img_width * img_height];

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

    // loop that determines which bin locations contain the dominant orientations
    for (int i = 0; i < sift::N_BINS; i++) {
        if (hist[i] >= ori_thresh * ori_max) {
            float prev = hist[(i - 1 + sift::N_BINS) % sift::N_BINS];
            float next = hist[(i + 1) % sift::N_BINS];
            if (prev > hist[i] || next > hist[i]) {
                continue;
            }

            float theta = 2.0f*sift::M_PIf*(i+1.0f)/sift::N_BINS 
                          + sift::M_PIf/sift::N_BINS*(prev-next)/(prev-2.0f*hist[i]+next);

            // store dominant orientations in the output array
            orientations[tid * sift::N_BINS + i] = theta;
        }
    }
}


/* Summary:
 *     Helper function for the 'generate_descriptors' kernel which updates the
 *     descriptor histogram at a given location x,y
 * Parameters:
 *   - hist: the descriptor histogram to modify
 *   - x: normalized pixel x-coordinate which will contribute to the histogram
 *   - y: normalized pixel y-coordinate which will contribute to the histogram
 *   - contrib: gradient value contribution to the histogram
 *   - theta_mn: orientaiton for location m,n within pixel neighborhood
 *   - lambda_desc: bounding variable for pixel distance from neighborhood edge
 * Return:
 *     (void) modifies the values in the 'hist' array
 */
__device__ void update_histograms_device(float* hist,
                                         float x,
                                         float y,
                                         float contrib,
                                         float theta_mn,
                                         float lambda_desc)
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


/* Summary:
 *     Helper function for the 'generate_descriptors' that turns the descriptor
 *     histograms into a descriptor (128-value vector)
 * Parameters:
 *   - histograms: input array of histograms that hold descriptor orientations
 *   - feature_vec: output array to store the keypoint descriptor
 * Return:
 *     (void) modifies the values in the 'feature_vec' array
 */
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


/* Summary:
 *     Kernel that generates the descriptors for a list of keypoints
 * Parameters:
 *   - gradPyramid: input array of gradient data for all images in scale space.
 *                  Concatenated [gx][gy] data for each image in a 1D array,
 *                  i.e. [img1_gx][img1_gy][img2_gx][img2_gy]... where img_gx
 *                  is a 1D array of all x-direction gradient data
 *   - keypoints: input array of all keypoints to process
 *   - keypointDescriptors: output array where all descriptors are placed
 *   - thetas: input array of all dominant orientations for the keypoints
 *   - imgOffsets: input array holding the start index locations for all
 *                 images in the scale space 'gradPyramid'. This was necessary
 *                 bookkeeping for accessing the gradPyramid as a combined
 *                 1D array of images at all octaves and scales.
 *                 i.e. {img1_offset, img2_offset, img3_offset, ...}
 *   - imgWidths: input array holding the image widths for all images in the
 *                scale space array 'gradPyramid' 
 *                i.e. {img1_width, img2_width, img3_width, ...}
 *   - imgHeights: input array holding the image heights for all images in the
 *                 scale space array 'gradPyramid'
 *                 i.e. {img1_height, img2_height, img3_height, ...}
 *   - num_kps: number of keypoints to be processed
 *   - num_scales_per_octave: number of scales for each octave in 'gradPyramid'
 *   - lambda_desc: thresholds the keypoints for proximity to border
 * Return:
 *     (void) places generated keypoint descriptors into 'keypointDescriptors'
 * Notes:
 *     Granularity is one thread per keypoint. Each thread will generate the
 *     descriptor for a single keypoint.
 */
__global__ void generate_descriptors(float* gradPyramid,
                                     sift::Keypoint* keypoints,
                                     uint8_t* keypointDescriptors,
                                     float* thetas,
                                     int *imgOffsets,
                                     int *imgWidths,
                                     int *imgHeights,
                                     int num_kps,
                                     int num_scales_per_octave,
                                     float lambda_desc)
{
    // thread ID used to index into keypoint array
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_kps)
        return;

    sift::Keypoint kp = keypoints[tid];
    float theta = thetas[tid];
     // Get the image index that the keypoint belongs to
    int img_idx = kp.octave * num_scales_per_octave + kp.scale;
    // bookkeeping required to access correct location in gradPyramid
    int img_offset = imgOffsets[img_idx];
    int img_width = imgWidths[img_idx];
    int img_height = imgHeights[img_idx];
    float pix_dist = sift::MIN_PIX_DIST * powf(2.0f, kp.octave);
    float histograms[sift::N_HIST * sift::N_HIST * sift::N_ORI] = {0};

    float gx, gy, theta_mn, grad_norm, weight, contribution;

    // determines the pixel neighborhood used in descriptor generation
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

            // get the X and Y gradient values from the gradPyramid
            gx = gradPyramid[img_offset + idx];
            gy = gradPyramid[img_offset + idx + img_width * img_height];
            grad_norm = sqrtf(gx * gx + gy * gy);

            theta_mn = fmodf(atan2f(gy, gx) - theta + 4.0f * sift::M_PIf, 2.0f * sift::M_PIf);
            weight = expf(-(powf(m * pix_dist - kp.x, 2.0f) + powf(n * pix_dist - kp.y, 2.0f)) / (2.0f * patch_sigma * patch_sigma));
            contribution = weight * grad_norm;

            update_histograms_device(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // convert the descriptor histogram to a 128-element vector which is
    // the keypoint descriptor format
    hists_to_vec_device(histograms, keypointDescriptors + tid * 128);
}


/* Summary:
 *     Finds dominant orientations and generates keypoints in a single kernel.
 * Parameters:
 *   - gradPyramid: input array of gradient data for all images in scale space.
 *                  Concatenated [gx][gy] data for each image in a 1D array,
 *                  i.e. [img1_gx][img1_gy][img2_gx][img2_gy]... where img_gx
 *                  is a 1D array of all x-direction gradient data
 *   - keypoints: input array of all keypoints to process
 *   - keypointDescriptors: output array where all descriptors are placed
 *   - orientations: output array which will hold the output orientations
 *                   produced for each keypoint. The indices of this array
 *                   match the keypoint indices in 'keypoints'
 *   - imgOffsets: input array holding the start index locations for all
 *                 images in the scale space 'gradPyramid'. This was necessary
 *                 bookkeeping for accessing the gradPyramid as a combined
 *                 1D array of images at all octaves and scales.
 *                 i.e. {img1_offset, img2_offset, img3_offset, ...}
 *   - imgWidths: input array holding the image widths for all images in the
 *                scale space array 'gradPyramid' 
 *                i.e. {img1_width, img2_width, img3_width, ...}
 *   - imgHeights: input array holding the image heights for all images in the
 *                 scale space array 'gradPyramid'
 *                 i.e. {img1_height, img2_height, img3_height, ...}
 *   - num_kps: number of keypoints to be processed
 *   - num_scales_per_octave: number of scales for each octave in 'gradPyramid'
 *   - lambda_ori: controls size of orientation neighborhood for a keypoint
 *   - lambda_desc: thresholds the keypoints for proximity to border
 * Return:
 *     (void) places generated keypoint descriptors into 'keypointDescriptors'
 * Notes:
 *     Granularity is one thread per keypoint. Each thread will generate the
 *     orientations and descriptor for a single keypoint. This combined kernel
 *     ended up being slower than running the orientation and keypoint kernels
 *     separately which prompted a different route for optimizations.
 */
__global__ void generate_orientations_and_descriptors(float* gradPyramid,
                                                      sift::Keypoint* keypoints,
                                                      uint8_t* keypointDescriptors,
                                                      float *orientations,
                                                      int *imgOffsets,
                                                      int *imgWidths,
                                                      int *imgHeights,
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

    sift::Keypoint kp = keypoints[tid];
    // Get the image index that the keypoint belongs to
    int img_idx = kp.octave * num_scales_per_octave + kp.scale;
    int img_offset = imgOffsets[img_idx];
    int img_width = imgWidths[img_idx];
    int img_height = imgHeights[img_idx];
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

    // determine the pixel neighborhood for orientation generation
    int x_start = roundf((kp.x - patch_radius) / pix_dist);
    int x_end   = roundf((kp.x + patch_radius) / pix_dist);
    int y_start = roundf((kp.y - patch_radius) / pix_dist);
    int y_end   = roundf((kp.y + patch_radius) / pix_dist);

    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            int clamped_x = max(0, min(x, img_width - 1));
            int clamped_y = max(0, min(y, img_height - 1));
            int idx = clamped_y * img_width + clamped_x;

            // gets the X and Y gradient information from the scale space
            gx = gradPyramid[img_offset + idx];
            gy = gradPyramid[img_offset + idx + img_width * img_height];
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

    // based on the dominant orientations calculated above, generate the
    // descriptors for the keypoint
    float contribution, half_size, theta_mn;
    for (int i = 0; i < sift::N_BINS; i++) {
        if (hist[i] >= ori_thresh * ori_max) {
            float prev = hist[(i - 1 + sift::N_BINS) % sift::N_BINS];
            float next = hist[(i + 1) % sift::N_BINS];
            if (prev > hist[i] || next > hist[i]) {
                continue;
            }

            float theta = 2*sift::M_PIf*(i+1)/sift::N_BINS 
                            + sift::M_PIf/sift::N_BINS*(prev-next)/(prev-2*hist[i]+next);

            orientations[tid * sift::N_BINS + i] = theta;

            // determines the pixel neighborhood for generating the descriptor
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

                    gx = gradPyramid[img_offset + idx];
                    gy = gradPyramid[img_offset + idx + img_width * img_height];
                    
                    theta_mn = fmodf(atan2f(gy, gx) - theta + 4 * sift::M_PIf, 2 * sift::M_PIf);
                    grad_norm = sqrtf(gx * gx + gy * gy);
                    weight = expf(-(powf(m * pix_dist - kp.x, 2) + powf(n * pix_dist - kp.y, 2)) / (2 * patch_sigma * patch_sigma));
                    contribution = weight * grad_norm;

                    update_histograms_device(histograms, x, y, contribution, theta_mn, lambda_desc);
                }
            }

            // convert the descriptor histograms to a 128-element array which is the
            // final keypoint descriptor format
            hists_to_vec_device(histograms, keypointDescriptors + 128 * (tid * sift::N_BINS + i));
        }
    }
}


/* Summary:
 *     Helper function for the 'generate_descriptors_one_block_per_kp' kernel
 *     which updates the descriptor histogram at a given location x,y
 * Parameters:
 *   - histograms: the descriptor histogram to modify
 *   - x: normalized pixel x-coordinate which will contribute to the histogram
 *   - y: normalized pixel y-coordinate which will contribute to the histogram
 *   - contrib: gradient value contribution to the histogram
 *   - theta_mn: orientaiton for location m,n within pixel neighborhood
 *   - lambda_desc: bounding variable for pixel distance from neighborhood edge
 * Return:
 *     (void) modifies the values in the 'histograms' array
 * Notes:
 *     Uses an atomicAdd since the 'generate_descriptors_one_block_per_kp' will
 *     have multiple threads working to generate descriptors for a single
 *     keypoint, and therefore multiple threads may try to update the same bin.
 */
__device__ void update_histograms_device_atomic(float* histograms,
                                                float x_rotated,
                                                float y_rotated,
                                                float contrib,
                                                float theta_mn,
                                                float lambda_desc)
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
                 float theta_diff = fmodf(theta_k - theta_mn + 2.0f * sift::M_PIf, 2.0f * sift::M_PIf);

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


/* Summary:
 *     Kernel that generates the descriptors for a list of keypoints
 * Parameters:
 *   - gradPyramid: input array of gradient data for all images in scale space.
 *                  Concatenated [gx][gy] data for each image in a 1D array,
 *                  i.e. [img1_gx][img1_gy][img2_gx][img2_gy]... where img_gx
 *                  is a 1D array of all x-direction gradient data
 *   - keypoints: input array of all keypoints to process
 *   - keypointDescriptors: output array where all descriptors are placed
 *   - thetas: input array of all dominant orientations for the keypoints
 *   - imgOffsets: input array holding the start index locations for all
 *                 images in the scale space 'gradPyramid'. This was necessary
 *                 bookkeeping for accessing the gradPyramid as a combined
 *                 1D array of images at all octaves and scales.
 *                 i.e. {img1_offset, img2_offset, img3_offset, ...}
 *   - imgWidths: input array holding the image widths for all images in the
 *                scale space array 'gradPyramid' 
 *                i.e. {img1_width, img2_width, img3_width, ...}
 *   - imgHeights: input array holding the image heights for all images in the
 *                 scale space array 'gradPyramid'
 *                 i.e. {img1_height, img2_height, img3_height, ...}
 *   - num_kps: number of keypoints to be processed
 *   - num_scales_per_octave: number of scales for each octave in 'gradPyramid'
 *   - lambda_desc: thresholds the keypoints for proximity to border
 * Return:
 *     (void) places generated keypoint descriptors into 'keypointDescriptors'
 * Notes:
 *     Granularity is one block per keypoint. The workload of analyzing the
 *     pixel neighborhood around a keypoint is divided among the threads
 *     within a single block. This allows for finer-grained parallelism
 *     and makes this faster than the naive 'generate_descriptors' kernel
 */
__global__ void generate_descriptors_one_block_per_kp(float* gradPyramid,
                                                      sift::Keypoint* keypoints,
                                                      uint8_t* keypointDescriptors,
                                                      float* thetas,
                                                      int *imgOffsets,
                                                      int *imgWidths,
                                                      int *imgHeights,
                                                      int num_kps,
                                                      int num_scales_per_octave,
                                                      float lambda_desc)
{
    // use the block index to map to keypoints 
    int bIdx = blockIdx.x;

    if (bIdx >= num_kps)
        return;

    sift::Keypoint kp = keypoints[bIdx];
    float theta = thetas[bIdx];
    int img_idx = kp.octave * num_scales_per_octave + kp.scale;
    int img_offset = imgOffsets[img_idx];
    int img_width = imgWidths[img_idx];
    int img_height = imgHeights[img_idx];
    float pix_dist = sift::MIN_PIX_DIST * pow(2.0f, kp.octave);

    // allocate a constant amount of shared memory to hold the
    // descriptor histogram
    __shared__ float histograms[sift::DESC_HIST_SHARED_SIZE];

    // use one thread to initialize the shared histogram
    if (threadIdx.x == 0) {
        for(int i = 0; i < sift::DESC_HIST_SHARED_SIZE; ++i) {
           histograms[i] = 0.0f;
       }
    }
    __syncthreads();


    float cos_t = cosf(theta), sin_t = sinf(theta);
    float patch_sigma = lambda_desc * kp.sigma;

    // determines the neighborhood around the keypoint used to calculate the descriptor
    float half_size = sqrtf(2.0f) * lambda_desc * kp.sigma * (sift::N_HIST + 1.0f) / sift::N_HIST;
    int x_start_patch = roundf((kp.x - half_size) / pix_dist);
    int x_end_patch = roundf((kp.x + half_size) / pix_dist);
    int y_start_patch = roundf((kp.y - half_size) / pix_dist);
    int y_end_patch = roundf((kp.y + half_size) / pix_dist);

    int patch_width = x_end_patch - x_start_patch + 1;
    int patch_height = y_end_patch - y_start_patch + 1;
    int total_neighbors = patch_width * patch_height;

    // iterate through all neighbors and divide up the workload between threads in the block
    for (int neighbor_idx = threadIdx.x; neighbor_idx < total_neighbors; neighbor_idx += blockDim.x)
    {
        int m = x_start_patch + (neighbor_idx % patch_width);
        int n = y_start_patch + (neighbor_idx / patch_width);

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

        // get the X and Y gradient information from the scale space
        float gx = gradPyramid[img_offset + idx];
        float gy = gradPyramid[img_offset + idx + img_width * img_height];
        float grad_norm = sqrtf(gx * gx + gy * gy);

        float weight = expf(-(powf(m * pix_dist - kp.x, 2.0f) + powf(n * pix_dist - kp.y, 2.0f)) / (2.0f * patch_sigma * patch_sigma));
        float contribution = weight * grad_norm;
        float theta_mn = fmodf(atan2f(gy, gx) - theta + 4.0f * sift::M_PIf, 2.0f * sift::M_PIf);

        // write gradient information to the shared descriptor histogram
        update_histograms_device_atomic((float*) histograms, x, y, contribution, theta_mn, lambda_desc);

    }

    __syncthreads();

    if (threadIdx.x == 0) {
        // convert the histogram to a 128-element keypoint descriptor array
        hists_to_vec_device((float*) histograms, keypointDescriptors + bIdx * 128);
    }
}