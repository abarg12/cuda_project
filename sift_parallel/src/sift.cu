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


/*****************************************************************************/
/************************* SECTION 1: SERIAL FUNCTIONS ***********************/ 

ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = sqrtf(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = powf(2, 1.0f/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * powf(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(sqrtf(sigma_total*sigma_total - sigma_prev*sigma_prev));
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
                         -pyramid.octaves[i][j].get_pixel(x-1, y, 0)) * 0.5f;
                    grad.set_pixel(x, y, 0, gx);
                    gy = (pyramid.octaves[i][j].get_pixel(x, y+1, 0)
                         -pyramid.octaves[i][j].get_pixel(x, y-1, 0)) * 0.5f;
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
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3.0f;
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
    float pix_dist = MIN_PIX_DIST * powf(2.0f, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = fminf(fminf(kp.x, kp.y), 
                                       fminf(pix_dist*img_grad.width-kp.x,
                                             pix_dist*img_grad.height-kp.y));
    if (min_dist_from_border <= sqrtf(2.0f)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3.0f * patch_sigma;
    int x_start = roundf((kp.x - patch_radius)/pix_dist);
    int x_end = roundf((kp.x + patch_radius)/pix_dist);
    int y_start = roundf((kp.y - patch_radius)/pix_dist);
    int y_end = roundf((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = sqrtf(gx*gx + gy*gy);
            weight = expf(-(powf(x*pix_dist-kp.x, 2.0f)+powf(y*pix_dist-kp.y, 2.0f))
                              /(2.0f*patch_sigma*patch_sigma));
            theta = fmodf(atan2f(gy, gx)+2.0f*M_PIf, 2.0f*M_PIf);
            bin = (int)roundf(N_BINS/(2.0f*M_PIf)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8f, ori_max = 0.0f;
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
            float theta = 2.0f*M_PIf*(j+1.0f)/N_BINS + M_PIf/N_BINS*(prev-next)/(prev-2.0f*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}


void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1.0f+(float)N_HIST)/2.0f) * 2.0f*lambda_desc/N_HIST;
        if (fabsf(x_i-x) > 2.0f*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1.0f+(float)N_HIST)/2.0f) * 2.0f*lambda_desc/N_HIST;
            if (fabsf(y_j-y) > 2.0f*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1.0f - N_HIST*0.5f/lambda_desc*fabsf(x_i-x))
                               *(1.0f - N_HIST*0.5f/lambda_desc*fabsf(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2.0f*M_PIf*(k-1.0f)/N_ORI;
                float theta_diff = fmodf(theta_k-theta_mn+2.0f*M_PIf, 2.0f*M_PIf);
                if (fabsf(theta_diff) >= 2.0f*M_PIf/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5f/M_PIf*fabsf(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = sqrtf(norm);
    float norm2 = 0.0f;
    for (int i = 0; i < size; i++) {
        hist[i] = fminf(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = sqrtf(norm2);
    for (int i = 0; i < size; i++) {
        float val = floorf(512.0f*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * powf(2.0f, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = sqrtf(2.0f)*lambda_desc*kp.sigma*(N_HIST+1.0f)/N_HIST;
    int x_start = roundf((kp.x-half_size) / pix_dist);
    int x_end = roundf((kp.x+half_size) / pix_dist);
    int y_start = roundf((kp.y-half_size) / pix_dist);
    int y_end = roundf((kp.y+half_size) / pix_dist);

    float cos_t = cosf(theta), sin_t = sinf(theta);
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
            if (fmaxf(fabsf(x), fabsf(y)) > lambda_desc*(N_HIST+1.0f)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = fmodf(atan2f(gy, gx)-theta+4.0f*M_PIf, 2.0f*M_PIf);
            float grad_norm = sqrtf(gx*gx + gy*gy);
            float weight = expf(-(powf(m*pix_dist-kp.x, 2.0f)+powf(n*pix_dist-kp.y, 2.0f))
                                    /(2.0f*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
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



/*****************************************************************************/
/******************* SECTION 2: NAIVE CUDA HOST CODE *************************/

// Identify the location of keypoints within the DoG processed images
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

            CUDA_CHECK(cudaGetLastError());

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


    dim3 blockDim(256);
    dim3 gridDim((kps.size() + blockDim.x - 1) / blockDim.x);
    generate_orientations<<<gridDim, blockDim>>>(devicePyramid,
                                                       deviceKeypoints,
                                                       deviceOrientations,
                                                       deviceImgOffsets,
                                                       deviceImgWidths,
                                                       deviceImgHeights,
                                                       kps.size(),
                                                       grad_pyramid.imgs_per_octave,
                                                       lambda_ori,
                                                       lambda_desc);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaMemcpy(hostOrientations, deviceOrientations, kps.size() * N_BINS * sizeof(float), cudaMemcpyDeviceToHost));

    // create output std::vector of keypoint orientations
    std::vector<std::vector<float>> orientation_outvec;
    for (int i = 0; i < kps.size(); i++) {
        orientation_outvec.push_back({});
        for (int j = 0; j < N_BINS; j++) {
            if (hostOrientations[i * N_BINS + j] != 0.0) {
                orientation_outvec[i].push_back(hostOrientations[i * N_BINS + j]);
                // std::cout << "naive ori: " << hostOrientations[i * N_BINS + j] << "\n";
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
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypoints, kps.size() * sizeof(Keypoint)));
    Keypoint *hostKeypoints = kps.data();
    // Copy keypoints from host to device memory
    CUDA_CHECK(
        cudaMemcpy(deviceKeypoints, hostKeypoints, 
                   kps.size() * sizeof(Keypoint), cudaMemcpyHostToDevice));

    // Allocate device memory for each keypoint's descriptor
    uint8_t *deviceKeypointDescriptors;
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


    dim3 blockDim(256);
    dim3 gridDim((kps.size() + blockDim.x - 1) / blockDim.x);
    generate_descriptors<<<gridDim, blockDim>>>(devicePyramid,
                                                       deviceKeypoints,
                                                       deviceKeypointDescriptors,
                                                       deviceThetas,
                                                       deviceImgOffsets,
                                                       deviceImgWidths,
                                                       deviceImgHeights,
                                                       kps.size(),
                                                       grad_pyramid.imgs_per_octave,
                                                       lambda_desc);

    CUDA_CHECK(cudaGetLastError());

    // Copy the descriptors back from the device
    for (size_t i = 0; i < kps.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(kps[i].descriptor.data(), deviceKeypointDescriptors + i * 128,
                             128 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(devicePyramid));
    CUDA_CHECK(cudaFree(deviceKeypoints));
    CUDA_CHECK(cudaFree(deviceKeypointDescriptors));
    CUDA_CHECK(cudaFree(deviceThetas));
    CUDA_CHECK(cudaFree(deviceImgOffsets));
    CUDA_CHECK(cudaFree(deviceImgWidths));
    CUDA_CHECK(cudaFree(deviceImgHeights));
}


/*****************************************************************************/
/****************** SECTION 3: OPTIMIZED CUDA HOST CODE **********************/

// Identify the location of keypoints within the DoG processed images
std::vector<Keypoint> find_keypoints_tiled(const ScaleSpacePyramid& dog_pyramid,
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


            dim3 blockDim(8, 8);
            dim3 gridDim((img.width + blockDim.x - 1) / blockDim.x,
                        (img.height + blockDim.y - 1) / blockDim.y);

            int tile_size = blockDim.x * blockDim.y;
            int sharedSize = 3 * tile_size * sizeof(float); // for 3 scales

            identify_keypoints_tiled<<<gridDim, blockDim, sharedSize>>>(deviceImage,
                                                                        deviceKeypointOutput,
                                                                        img.size,
                                                                        img.width,
                                                                        img.height,
                                                                        contrast_thresh);

            CUDA_CHECK(cudaGetLastError());

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

std::vector<Keypoint> find_ori_desc_parallel_opt(std::vector<Keypoint>& kps,
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


    dim3 blockDim(256);
    dim3 gridDim((kps.size() + blockDim.x - 1) / blockDim.x);
    generate_orientations<<<gridDim, blockDim>>>(devicePyramid,
                                                       deviceKeypoints,
                                                       deviceOrientations,
                                                       deviceImgOffsets,
                                                       deviceImgWidths,
                                                       deviceImgHeights,
                                                       kps.size(),
                                                       grad_pyramid.imgs_per_octave,
                                                       lambda_ori,
                                                       lambda_desc);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaMemcpy(hostOrientations, deviceOrientations, kps.size() * N_BINS * sizeof(float), cudaMemcpyDeviceToHost));


    // get size of new vector of nonzero orientations and their keypoints
    int out_size = 0;
    for (int i = 0; i < kps.size(); i++) {
        for (int j = 0; j < N_BINS; j++) {
            if (hostOrientations[i * N_BINS + j] != 0.0) {
                out_size++;
            }
        }
    }

    // alocate memory for new vector of keypoints and orientations
    Keypoint *deviceKeypointsNew;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypointsNew, out_size * sizeof(Keypoint)));

    float *deviceOrientationsNew;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceOrientationsNew, out_size * sizeof(float)));
    CUDA_CHECK(
        cudaMemset(deviceOrientationsNew, 0, out_size * sizeof(float)));


    // Allocate device memory for each keypoint's descriptor
    uint8_t *deviceKeypointDescriptors;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypointDescriptors, out_size * 128 * sizeof(uint8_t)));
    // Zero out device memory
    CUDA_CHECK(
        cudaMemset(deviceKeypointDescriptors, 0, out_size * 128 * sizeof(uint8_t)));


    std::vector<Keypoint> kps_out;

    int new_arr_index = 0;
    for (int i = 0; i < kps.size(); i++) {
        for (int j = 0; j < N_BINS; j++) {
            if (hostOrientations[i * N_BINS + j] != 0.0) {
                cudaMemcpy(deviceKeypointsNew + new_arr_index,
                            deviceKeypoints + i,
                            sizeof(Keypoint), cudaMemcpyDeviceToDevice);
                cudaMemcpy(deviceOrientationsNew + new_arr_index,
                            hostOrientations + i * N_BINS + j,
                            sizeof(float), cudaMemcpyHostToDevice);
                Keypoint kp = kps[i];
                kps_out.push_back(kp);
                new_arr_index++;
            }
        }
    }

    // dim3 blockDimDesc(256);
    // dim3 gridDimDesc((out_size + blockDimDesc.x - 1) / blockDimDesc.x);
    // generate_descriptors<<<blockDimDesc, gridDimDesc>>>(devicePyramid,
    //                                                    deviceKeypointsNew,
    //                                                    deviceKeypointDescriptors,
    //                                                    deviceOrientationsNew,
    //                                                    deviceImgOffsets,
    //                                                    deviceImgWidths,
    //                                                    deviceImgHeights,
    //                                                    out_size,
    //                                                    grad_pyramid.imgs_per_octave,
    //                                                    lambda_desc);

    dim3 blockDimDesc(128); // Example block size, choose based on testing and workload
    dim3 gridDimDesc(out_size); // One block per filtered keypoint/orientation

    // Shared memory size is NOT specified for static shared memory
    generate_descriptors_one_block_per_kp<<<gridDimDesc, blockDimDesc>>>(devicePyramid,
                                                                         deviceKeypointsNew,
                                                                         deviceKeypointDescriptors,
                                                                         deviceOrientationsNew,
                                                                         deviceImgOffsets,
                                                                         deviceImgWidths,
                                                                         deviceImgHeights,
                                                                         out_size,
                                                                         grad_pyramid.imgs_per_octave,
                                                                         lambda_desc);

    CUDA_CHECK(cudaGetLastError());


    // Copy the descriptors back from the device
    for (size_t i = 0; i < out_size; ++i) {
        CUDA_CHECK(cudaMemcpy(kps_out[i].descriptor.data(), deviceKeypointDescriptors + i * 128,
                             128 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }


    CUDA_CHECK(cudaFree(devicePyramid));
    CUDA_CHECK(cudaFree(deviceKeypoints));
    CUDA_CHECK(cudaFree(deviceKeypointsNew));
    CUDA_CHECK(cudaFree(deviceKeypointDescriptors));
    CUDA_CHECK(cudaFree(deviceOrientations));
    CUDA_CHECK(cudaFree(deviceOrientationsNew));
    CUDA_CHECK(cudaFree(deviceImgOffsets));
    CUDA_CHECK(cudaFree(deviceImgWidths));
    CUDA_CHECK(cudaFree(deviceImgHeights));
    delete[] hostOrientations;

    return kps_out;
}



std::vector<Keypoint> find_ori_desc_parallel_combined(std::vector<Keypoint>& kps,
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
    uint8_t *deviceKeypointDescriptors;
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypoints, kps.size() * sizeof(Keypoint)));
    Keypoint *hostKeypoints = kps.data();
    // Copy keypoints from host to device memory
    CUDA_CHECK(
        cudaMemcpy(deviceKeypoints, hostKeypoints, 
                   kps.size() * sizeof(Keypoint), cudaMemcpyHostToDevice));

    // Allocate device memory for each keypoint's descriptor
    // Keypoints may have up to N_BINS descriptors each, in which case the keypoint is duplicated
    // for each valid descriptor
    CUDA_CHECK(
        cudaMalloc((void **) &deviceKeypointDescriptors, kps.size() * 128 * N_BINS * sizeof(uint8_t)));
    // Zero out device memory
    CUDA_CHECK(
        cudaMemset(deviceKeypointDescriptors, 0, kps.size() * 128 * N_BINS * sizeof(uint8_t)));


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


    dim3 blockDim(256);
    dim3 gridDim((kps.size() + blockDim.x - 1) / blockDim.x);
    generate_orientations_and_descriptors<<<gridDim, blockDim>>>(devicePyramid,
                                                       deviceKeypoints,
                                                       deviceKeypointDescriptors,
                                                       deviceOrientations,
                                                       deviceImgOffsets,
                                                       deviceImgWidths,
                                                       deviceImgHeights,
                                                       kps.size(),
                                                       grad_pyramid.imgs_per_octave,
                                                       lambda_ori,
                                                       lambda_desc);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaMemcpy(hostOrientations, deviceOrientations, kps.size() * N_BINS * sizeof(float), cudaMemcpyDeviceToHost));

    // create output std::vector of keypoints with descriptors
    std::vector<Keypoint> kps_out;
    for (int i = 0; i < kps.size(); i++) {
        for (int j = 0; j < N_BINS; j++) {
            if (hostOrientations[i * N_BINS + j] != 0.0) {
                Keypoint kp = kps[i];
                CUDA_CHECK(cudaMemcpy(kp.descriptor.data(),
                                      deviceKeypointDescriptors + 128 * (i * N_BINS + j),
                                      128 * sizeof(uint8_t),
                                      cudaMemcpyDeviceToHost));
                kps_out.push_back(kp);
            }
        }
    }

    CUDA_CHECK(cudaFree(devicePyramid));
    CUDA_CHECK(cudaFree(deviceKeypoints));
    CUDA_CHECK(cudaFree(deviceKeypointDescriptors));
    CUDA_CHECK(cudaFree(deviceOrientations));
    CUDA_CHECK(cudaFree(deviceImgOffsets));
    CUDA_CHECK(cudaFree(deviceImgWidths));
    CUDA_CHECK(cudaFree(deviceImgHeights));
    delete[] hostOrientations;

    return kps_out;
}


/*****************************************************************************/
/***** SECTION 4: SIFT 'MAIN' FUNCTION (find_keypoints_and_descriptors) ******/

/*
 * The serial 'main' function which calls all necessary functions to compute the
 * keypoints and descriptors.
 */
std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);

    printf("--------------------------------------------------\n");
    printf("RUNNING SERIAL SIFT\n");
    float total_time = 0.0;

    // Generate the Gaussian Pyramid
    auto t_start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("- Gaussian Pyramid elapsed time: %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

    // Generate the DoG Pyramid
    t_start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("- DoG Pyramid elapsed time: %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

    // Find keypoints
    t_start = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("- Find Keypoints elapsed time: %f ms\n", elapsed_ms);
    total_time += elapsed_ms;
    
    // Generate gradient pyramid
    t_start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("- Gradient Pyramid elapsed time: %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

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
    printf("- Keypoint Descriptor elapsed time: %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

    printf("Total runtime: %f ms\n\n", total_time);

    return kps;
}


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
    int kSize = ceilf(6.0f * sigma);
    if (kSize % 2 == 0) ++kSize;
    int kCenter = kSize / 2;
    std::vector<float> h_kernel(kSize);
    float sum = 0.0f;
    for (int k = -kSize/2; k <= kSize/2; k++) {
        float v = expf(-(k*k) / (2.0f * sigma * sigma));
        h_kernel[kCenter + k] = v; 
        sum += v;
    }
    for (int k = 0; k < kSize; k++) {
        h_kernel[k] /= sum;
    }

    // Allocate device memory
    float *d_in, *d_tmp, *d_out, *d_kernel;
    size_t imgBytes = width * height * sizeof(float);
    size_t kernelBytes = kSize * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_in,    imgBytes));
    CUDA_CHECK(cudaMalloc(&d_tmp,   imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out,   imgBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel,kernelBytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in,     img.data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernelBytes, cudaMemcpyHostToDevice));

    // Launch parameters
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1)/block.x,
              (height+ block.y - 1)/block.y);

    // Row blur
    gaussianBlurRow<<<grid, block>>>(d_in, d_tmp, width, height, d_kernel, kSize, kCenter);
    CUDA_CHECK(cudaGetLastError());

    // Column blur
    gaussianBlurCol<<<grid, block>>>(d_tmp, d_out, width, height, d_kernel, kSize, kCenter);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    Image filtered(width, height, 1);
    CUDA_CHECK(cudaMemcpy(filtered.data, d_out, imgBytes, cudaMemcpyDeviceToHost));

    // Free
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    cudaFree(d_kernel);

    return filtered;
}



ScaleSpacePyramid generate_gaussian_pyramid_parallel(
    const Image& img, float sigma_min,
    int num_octaves, int scales_per_octave)
{
    // same setup as serial
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = sqrtf(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur_gpu(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;
    float k = powf(2, 1.0f/scales_per_octave);
    std::vector<float> sigma_vals(imgs_per_octave);
    sigma_vals[0] = base_sigma;
    for (int i = 1; i < imgs_per_octave; i++) {
        float prev = base_sigma * powf(k, i-1);
        float total = k * prev;
        sigma_vals[i] = sqrtf(total*total - prev*prev);
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

ScaleSpacePyramid generate_dog_pyramid_parallel(
    const ScaleSpacePyramid& img_pyramid
) {
    int num_octaves    = img_pyramid.num_octaves;
    int levels_per_oct = img_pyramid.imgs_per_octave;
    
    ScaleSpacePyramid dog_pyramid = {
        num_octaves,
        levels_per_oct - 1,
        std::vector<std::vector<Image>>(num_octaves)
    };

    for (int o = 0; o < num_octaves; ++o) {
        // all images in this octave share width/height
        const Image& first = img_pyramid.octaves[o][0];
        int W = first.width, H = first.height;
        int N = W * H;

        // reserve space
        auto& dst = dog_pyramid.octaves[o];
        dst.reserve(levels_per_oct - 1);

        // allocate device buffers once per octave
        float *d_curr, *d_prev, *d_out;
        size_t bytes = N * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_curr, bytes));
        CUDA_CHECK(cudaMalloc(&d_prev, bytes));
        CUDA_CHECK(cudaMalloc(&d_out,  bytes));

        // set up launch geometry
        int threads = 256;
        int blocks  = (N + threads - 1) / threads;

        for (int s = 1; s < levels_per_oct; ++s) {
            const Image& curr = img_pyramid.octaves[o][s];
            const Image& prev = img_pyramid.octaves[o][s-1];

            // copy just these two images
            CUDA_CHECK(cudaMemcpy(d_curr, curr.data, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_prev, prev.data, bytes, cudaMemcpyHostToDevice));

            // run DoG
            computeDoG<<<blocks,threads>>>(d_curr, d_prev, d_out, N);
            CUDA_CHECK(cudaGetLastError());

            // pull result back
            Image diff(W, H, curr.channels);
            CUDA_CHECK(cudaMemcpy(diff.data, d_out, bytes, cudaMemcpyDeviceToHost));

            dst.push_back(std::move(diff));
        }

        // free per-octave
        cudaFree(d_curr);
        cudaFree(d_prev);
        cudaFree(d_out);
    }

    return dog_pyramid;
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


ScaleSpacePyramid generate_gradient_pyramid_parallel(
    const ScaleSpacePyramid& pyramid
) {
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };

    // one 16×16 block covers 256 threads:
    dim3 block(16,16);

    for (int o = 0; o < pyramid.num_octaves; ++o) {
      grad_pyramid.octaves[o].reserve(pyramid.imgs_per_octave);

      for (int s = 0; s < pyramid.imgs_per_octave; ++s) {
        const Image& src = pyramid.octaves[o][s];
        int W = src.width, H = src.height;
        // allocate device buffers
        float *d_in, *d_out;
        size_t in_bytes  = W*H*sizeof(float);
        size_t out_bytes = W*H*2*sizeof(float);
        CUDA_CHECK( cudaMalloc(&d_in,  in_bytes) );
        CUDA_CHECK( cudaMalloc(&d_out, out_bytes) );

        // copy input
        CUDA_CHECK( cudaMemcpy(d_in, src.data, in_bytes, cudaMemcpyHostToDevice) );

        // launch
        dim3 grid( (W+block.x-1)/block.x,
                   (H+block.y-1)/block.y );
        gradientKernel<<<grid,block>>>(d_in, d_out, W, H);
        CUDA_CHECK( cudaGetLastError() );

        // copy back into a 2‐channel Image
        Image grad(W, H, 2);
        CUDA_CHECK( cudaMemcpy(grad.data, d_out, out_bytes, cudaMemcpyDeviceToHost) );

        // free device
        cudaFree(d_in);
        cudaFree(d_out);

        grad_pyramid.octaves[o].push_back(std::move(grad));
      }
    }

    return grad_pyramid;
}


/*
 * The parallel 'main' function which calls all necessary functions to compute
 * the keypoints and descriptors. This calls the CUDA parallel implementations.
 */
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
    float elapsed_ms;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    printf("--------------------------------------------------\n");
    printf("RUNNING PARALLEL SIFT (NAIVE)\n");
    float total_time = 0.0;

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);

    // Generate the Gaussian Pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid_parallel(input, sigma_min, num_octaves,
                                                                            scales_per_octave);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    printf("- Gaussian Pyramid elapsed time %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

    // Generate the DoG Pyramid
    cudaEventRecord(startEvent, 0);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid_parallel(gaussian_pyramid);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    printf("- DoG Pyramid elapsed time %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

    // Find keypoints
    cudaEventRecord(startEvent, 0);
    std::vector<Keypoint> tmp_kps = find_keypoints_parallel_naive(dog_pyramid, contrast_thresh, edge_thresh);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    printf("- Find Keypoints elapsed time %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

    // Generate gradient pyramid
    cudaEventRecord(startEvent, 0);
    // This parallel code does not pass the accuracy verification
    // ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid_parallel(gaussian_pyramid);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    // printf("- generate_gradient_pyramid elapsed time %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

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
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    printf("- Keypoint Descriptor elapsed time %f ms\n", elapsed_ms);
    total_time += elapsed_ms;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    printf("Total runtime: %f ms\n", total_time);

    printf("\n[omitted] parallel Gradient Pyramid\n\n");

    return kps;
}
// BELOW IS OPTIMIZED PARALLEL FUNCTIONS

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

// Persistent device buffers for optimized blur
static float *d_in_opt = nullptr, *d_tmp_opt = nullptr, *d_out_opt = nullptr, *d_kernel_opt = nullptr;
static int   capPixels_opt = 0, capKSize_opt = 0;

// Optimized GPU-based Gaussian blur
Image gaussian_blur_gpu_optimized(const Image& img, float sigma)
{
    assert(img.channels == 1);
    int W = img.width, H = img.height;
    int pixels = W * H;

    // Build 1D Gaussian kernel on host
    int kSize = (int)ceilf(6.0f * sigma);
    if (kSize % 2 == 0) ++kSize;
    int kCenter = kSize / 2;
    std::vector<float> h_kernel(kSize);
    float sum = 0.0f, inv2s = 1.0f / (2.0f * sigma * sigma);
    for (int i = -kCenter; i <= kCenter; ++i) {
        float v = expf(-i * i * inv2s);
        h_kernel[i + kCenter] = v;
        sum += v;
    }
    for (auto &v : h_kernel) v /= sum;

    // (Re)allocate pixel buffers if needed
    if (pixels > capPixels_opt) {
        if (d_in_opt) {
            cudaFree(d_in_opt);
            cudaFree(d_tmp_opt);
            cudaFree(d_out_opt);
        }
        CUDA_CHECK(cudaMalloc(&d_in_opt,  pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp_opt, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_opt, pixels * sizeof(float)));
        capPixels_opt = pixels;
    }

    // (Re)allocate kernel buffer if needed
    if (kSize > capKSize_opt) {
        if (d_kernel_opt) cudaFree(d_kernel_opt);
        CUDA_CHECK(cudaMalloc(&d_kernel_opt, kSize * sizeof(float)));
        capKSize_opt = kSize;
    }

    // Copy input image and kernel to device
    CUDA_CHECK(cudaMemcpy(d_in_opt,     img.data,       pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel_opt, h_kernel.data(), kSize * sizeof(float),    cudaMemcpyHostToDevice));

    // Launch parameters
    dim3 block(16,16), grid((W+15)/16, (H+15)/16);

    // Row pass
    gaussianBlurRow_optimized<<<grid, block>>>(d_in_opt, d_tmp_opt, W, H, d_kernel_opt, kSize, kCenter);
    CUDA_CHECK(cudaGetLastError());

    // Column pass
    gaussianBlurCol_optimized<<<grid, block>>>(d_tmp_opt, d_out_opt, W, H, d_kernel_opt, kSize, kCenter);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    Image filtered(W, H, 1);
    CUDA_CHECK(cudaMemcpy(filtered.data, d_out_opt, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    return filtered;
}

// Optimized SIFT Gaussian pyramid
ScaleSpacePyramid generate_gaussian_pyramid_optimized(
    const Image& img, float sigma_min,
    int num_octaves, int scales_per_octave)
{
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = sqrtf(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur_gpu_optimized(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;
    float k = powf(2.0f, 1.0f/scales_per_octave);
    std::vector<float> sigma_vals(imgs_per_octave);
    sigma_vals[0] = base_sigma;
    for (int i = 1; i < imgs_per_octave; ++i) {
        float p = base_sigma * powf(k, i-1);
        float t = p * k;
        sigma_vals[i] = sqrtf(t*t - p*p);
    }

    ScaleSpacePyramid pyramid{num_octaves, imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)};
    for (int o = 0; o < num_octaves; ++o) {
        auto& octave = pyramid.octaves[o];
        octave.reserve(imgs_per_octave);
        octave.push_back(std::move(base_img));
        for (int s = 1; s < imgs_per_octave; ++s) {
            const Image& prev = octave.back();
            octave.push_back(gaussian_blur_gpu_optimized(prev, sigma_vals[s]));
        }
        const Image& next_base = octave[imgs_per_octave - 3];
        base_img = next_base.resize(next_base.width/2, next_base.height/2, Interpolation::NEAREST);
    }
    return pyramid;
}

//DoG Pyramid

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

// Optimized DoG pyramid
ScaleSpacePyramid generate_dog_pyramid_optimized(
    const ScaleSpacePyramid& img_pyramid
) {
    int num_octaves    = img_pyramid.num_octaves;
    int levels_per_oct = img_pyramid.imgs_per_octave;

    ScaleSpacePyramid dog_pyramid{
        num_octaves,
        levels_per_oct - 1,
        std::vector<std::vector<Image>>(num_octaves)
    };

    // persistent device buffers
    static float *d_flat = nullptr, *d_out = nullptr;
    static int   cap_flat = 0, cap_out = 0;

    for (int o = 0; o < num_octaves; ++o) {
        auto& dst = dog_pyramid.octaves[o];
        dst.reserve(levels_per_oct - 1);

        const Image& L0 = img_pyramid.octaves[o][0];
        int W = L0.width, H = L0.height;
        int N = W * H;
        int totalIn  = N * levels_per_oct;
        int totalOut = N * (levels_per_oct - 1);

        // grow d_flat if needed
        if (totalIn > cap_flat) {
            if (d_flat) cudaFree(d_flat);
            CUDA_CHECK(cudaMalloc(&d_flat, totalIn * sizeof(float)));
            cap_flat = totalIn;
        }
        // grow d_out if needed
        if (totalOut > cap_out) {
            if (d_out) cudaFree(d_out);
            CUDA_CHECK(cudaMalloc(&d_out, totalOut * sizeof(float)));
            cap_out = totalOut;
        }

        // upload levels into flat buffer
        for (int s = 0; s < levels_per_oct; ++s) {
            const Image& img = img_pyramid.octaves[o][s];
            CUDA_CHECK(cudaMemcpy(
                d_flat + s * N,
                img.data,
                N * sizeof(float),
                cudaMemcpyHostToDevice
            ));
        }

        // launch unified DoG kernel
        int threads = 256;
        int blocks  = (totalOut + threads - 1) / threads;
        computeDoGAll<<<blocks,threads>>>(d_flat, d_out, N, levels_per_oct);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // download each diff image
        for (int s = 1; s < levels_per_oct; ++s) {
            Image diff(W, H, L0.channels);
            CUDA_CHECK(cudaMemcpy(
                diff.data,
                d_out + (s-1) * N,
                N * sizeof(float),
                cudaMemcpyDeviceToHost
            ));
            dst.push_back(std::move(diff));
        }
    }

    return dog_pyramid;
}

// Gradient_Pyramid

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

// Optimized gradient pyramid
ScaleSpacePyramid generate_gradient_pyramid_optimized(
    const ScaleSpacePyramid& pyramid)
{
    int num_octaves    = pyramid.num_octaves;
    int levels_per_oct = pyramid.imgs_per_octave;
    ScaleSpacePyramid grad_pyramid{
        num_octaves,
        levels_per_oct,
        std::vector<std::vector<Image>>(num_octaves)
    };

    // persistent buffers
    static float *d_in  = nullptr, *d_out = nullptr;
    static int   cap_in = 0, cap_out = 0;

    for (int o = 0; o < num_octaves; ++o) {
        const Image& L0 = pyramid.octaves[o][0];
        int W = L0.width, H = L0.height;
        int N = W * H;
        int totalIn  = levels_per_oct * N;
        int totalOut = 2 * totalIn;

        // grow d_in if needed
        if (totalIn > cap_in) {
            if (d_in) cudaFree(d_in);
            CUDA_CHECK(cudaMalloc(&d_in, totalIn * sizeof(float)));
            cap_in = totalIn;
        }
        // grow d_out if needed
        if (totalOut > cap_out) {
            if (d_out) cudaFree(d_out);
            CUDA_CHECK(cudaMalloc(&d_out, totalOut * sizeof(float)));
            cap_out = totalOut;
        }

        // upload all levels into d_in
        for (int s = 0; s < levels_per_oct; ++s) {
            const Image& img = pyramid.octaves[o][s];
            CUDA_CHECK(cudaMemcpy(
                d_in + s * N,
                img.data,
                N * sizeof(float),
                cudaMemcpyHostToDevice
            ));
        }

        // launch gradient kernel
        int threads = 256;
        int blocks  = (totalIn + threads - 1) / threads;
        gradientKernelAll<<<blocks, threads>>>(d_in, d_out, W, H, levels_per_oct, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // download and unpack
        std::vector<float> h_out(totalOut);
        CUDA_CHECK(cudaMemcpy(
            h_out.data(), d_out,
            totalOut * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        auto& dst = grad_pyramid.octaves[o];
        dst.reserve(levels_per_oct);
        for (int s = 0; s < levels_per_oct; ++s) {
            Image gradImg(W, H, 2);
            int base = 2 * s * N;
            for (int i = 0; i < N; ++i) {
                gradImg.data[2*i + 0] = h_out[base + 2*i + 0];
                gradImg.data[2*i + 1] = h_out[base + 2*i + 1];
            }
            dst.push_back(std::move(gradImg));
        }
    }

    return grad_pyramid;
}


/*
 * The parallel 'main' function which calls all necessary functions to compute
 * the keypoints and descriptors. This calls the optimized CUDA implementation.
 */
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
    cudaEvent_t startEvent, stopEvent, startTotalEvent, stopTotalEvent;
    float elapsedTime, totalTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&startTotalEvent);
    cudaEventCreate(&stopTotalEvent);

    printf("--------------------------------------------------\n");
    printf("RUNNING PARALLEL SIFT (OPTIMIZED)\n");
    cudaEventRecord(startTotalEvent, 0);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);

    cudaEventRecord(startEvent, 0);
    // Generate the Gaussian Pyramid
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid_optimized(input, sigma_min, num_octaves,
                                                                            scales_per_octave);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventRecord(stopTotalEvent, 0);
    cudaEventSynchronize(stopTotalEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("- gaussian pyramid elapsed time %f ms\n", elapsedTime);

    cudaEventRecord(startEvent, 0);
    // Generate the DoG Pyramid
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid_optimized(gaussian_pyramid);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventRecord(stopTotalEvent, 0);
    cudaEventSynchronize(stopTotalEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("- DoG pyramid elapsed time %f ms\n", elapsedTime);

    // Find keypoints
    // cudaEventRecord(startEvent, 0);
    // This optimized code fails the accuracy verifications
    // std::vector<Keypoint> tmp_kps = find_keypoints_tiled(dog_pyramid, contrast_thresh, edge_thresh);
    std::vector<Keypoint> tmp_kps = find_keypoints_parallel_naive(dog_pyramid, contrast_thresh, edge_thresh);
    // cudaEventRecord(stopEvent, 0);
    // cudaEventSynchronize(stopEvent);
    // cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    // printf("- find_keypoints_tiled elapsed time %f ms\n", elapsedTime);

    // Generate gradient pyramid
    // cudaEventRecord(startEvent, 0);
    // This optimized code fails the accuracy verifications
    // ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid_optimized(gaussian_pyramid);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    // cudaEventRecord(stopEvent, 0);
    // cudaEventSynchronize(stopEvent);
    // cudaEventRecord(stopTotalEvent, 0);
    // cudaEventSynchronize(stopTotalEvent);
    // cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    // printf("- Gradient pyramid elapsed time %f ms\n", elapsedTime);

    // Generate keypoint descriptors

    // This was V1 of the optimized kernel which ran slower than the naive parallel
    // std::vector<Keypoint> kps = find_ori_desc_parallel_combined(tmp_kps,
    //                                                     grad_pyramid,
    //                                                     lambda_ori,
    //                                                     lambda_desc);
    cudaEventRecord(startEvent, 0);
    std::vector<Keypoint> kps = find_ori_desc_parallel_opt(tmp_kps,
                                                        grad_pyramid,
                                                        lambda_ori,
                                                        lambda_desc);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("- keypoint descriptor generation elapsed time %f ms\n", elapsedTime);

    cudaEventRecord(stopTotalEvent, 0);
    cudaEventSynchronize(stopTotalEvent);
    cudaEventElapsedTime(&totalTime, startTotalEvent, stopTotalEvent);
    printf("Total runtime: %f ms\n", totalTime);

    printf("\n[omitted] optimized Find Keypoints\n");
    printf("[omitted] optimized Gradient Pyramid\n\n");


    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startTotalEvent);
    cudaEventDestroy(stopTotalEvent);

    return kps;
}


/*
 * Timing utility function that finds averaged timing results
 * for serial and parallel code run over 10 different images
 */
std::vector<std::vector<float>> find_keypoints_and_descriptors_timing(
    std::vector<Image> imgs,
    float sigma_min,
    int num_octaves,
    int scales_per_octave, 
    float contrast_thresh,
    float edge_thresh, 
    float lambda_ori,
    float lambda_desc)
{
    int phases = 6; // gaussian, DoG, kps, gradient pyramid, descriptors, total time
    int sift_types = 3; // serial, parallel naive, parallel optimized

    std::vector<std::vector<float>> timing_data(phases, std::vector<float>(sift_types, 0.0f));

    int num_imgs = imgs.size();

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    // Run and time serial SIFT
    float serial_gaus_time = 0.0f;
    float serial_dog_time = 0.0f;
    float serial_kps_time = 0.0f;
    float serial_grad_time = 0.0f;
    float serial_desc_time = 0.0f;
    float serial_total_time = 0.0f;
    for (int i = 0; i < num_imgs; i++) {
        start = std::chrono::high_resolution_clock::now();
        ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(imgs[i], sigma_min, num_octaves, scales_per_octave);
        end = std::chrono::high_resolution_clock::now();
        serial_gaus_time += std::chrono::duration<float, std::milli>(end - start).count();
        serial_total_time += std::chrono::duration<float, std::milli>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
        end = std::chrono::high_resolution_clock::now();
        serial_dog_time += std::chrono::duration<float, std::milli>(end - start).count();
        serial_total_time += std::chrono::duration<float, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
        end = std::chrono::high_resolution_clock::now();
        serial_kps_time += std::chrono::duration<float, std::milli>(end - start).count();
        serial_total_time += std::chrono::duration<float, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
        end = std::chrono::high_resolution_clock::now();
        serial_grad_time += std::chrono::duration<float, std::milli>(end - start).count();
        serial_total_time += std::chrono::duration<float, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
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
        end = std::chrono::high_resolution_clock::now();
        serial_desc_time += std::chrono::duration<float, std::milli>(end - start).count();
        serial_total_time += std::chrono::duration<float, std::milli>(end - start).count();
    }
    
    timing_data[0][0] = serial_gaus_time / num_imgs;
    timing_data[1][0] = serial_dog_time / num_imgs;
    timing_data[2][0] = serial_kps_time / num_imgs;
    timing_data[3][0] = serial_grad_time / num_imgs;
    timing_data[4][0] = serial_desc_time / num_imgs;
    timing_data[5][0] = serial_total_time / num_imgs;


    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float elapsed_ms;

    // Run and time naive parallel SIFT
    float naive_gaus_time = 0.0f;
    float naive_dog_time = 0.0f;
    float naive_kps_time = 0.0f;
    float naive_grad_time = 0.0f;
    float naive_desc_time = 0.0f;
    float naive_total_time = 0.0f;
    for (int i = 0; i < num_imgs; i++) {
        cudaEventRecord(startEvent, 0);
        ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid_parallel(imgs[i], sigma_min, num_octaves, scales_per_octave);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        naive_gaus_time += elapsed_ms;
        naive_total_time += elapsed_ms;

        cudaEventRecord(startEvent, 0);
        ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        naive_dog_time += elapsed_ms;
        naive_total_time += elapsed_ms;

        cudaEventRecord(startEvent, 0);
        std::vector<Keypoint> tmp_kps = find_keypoints_parallel_naive(dog_pyramid, contrast_thresh, edge_thresh);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        naive_kps_time += elapsed_ms;
        naive_total_time += elapsed_ms;

        cudaEventRecord(startEvent, 0);
        ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        naive_grad_time += elapsed_ms;
        naive_total_time += elapsed_ms;

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
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        naive_desc_time += elapsed_ms;
        naive_total_time += elapsed_ms;
    }

    timing_data[0][1] = naive_gaus_time / num_imgs;
    timing_data[1][1] = naive_dog_time / num_imgs;
    timing_data[2][1] = naive_kps_time / num_imgs;
    timing_data[3][1] = naive_grad_time / num_imgs;
    timing_data[4][1] = naive_desc_time / num_imgs;
    timing_data[5][1] = naive_total_time / num_imgs;


    // Run and time optimized parallel SIFT
    float opt_gaus_time = 0.0f;
    float opt_dog_time = 0.0f;
    float opt_kps_time = 0.0f;
    float opt_grad_time = 0.0f;
    float opt_desc_time = 0.0f;
    float opt_total_time = 0.0f;
    for (int i = 0; i < num_imgs; i++) {
        cudaEventRecord(startEvent, 0);
        ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid_optimized(imgs[i], sigma_min, num_octaves, scales_per_octave);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        opt_gaus_time += elapsed_ms;
        opt_total_time += elapsed_ms;

        cudaEventRecord(startEvent, 0);
        ScaleSpacePyramid dog_pyramid = generate_dog_pyramid_optimized(gaussian_pyramid);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        opt_dog_time += elapsed_ms;
        opt_total_time += elapsed_ms;

        cudaEventRecord(startEvent, 0);
        // This optimized code fails the accuracy verifications
        // std::vector<Keypoint> tmp_kps = find_keypoints_tiled(dog_pyramid, contrast_thresh, edge_thresh);
        std::vector<Keypoint> tmp_kps = find_keypoints_parallel_naive(dog_pyramid, contrast_thresh, edge_thresh);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        opt_kps_time += elapsed_ms;
        opt_total_time += elapsed_ms;

        cudaEventRecord(startEvent, 0);
        // This optimized code fails the accuracy verifications
        // ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid_optimized(gaussian_pyramid);
        ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        opt_grad_time += elapsed_ms;
        opt_total_time += elapsed_ms;

        cudaEventRecord(startEvent, 0);
        std::vector<Keypoint> kps = find_ori_desc_parallel_opt(tmp_kps, grad_pyramid, lambda_ori, lambda_desc);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        opt_desc_time += elapsed_ms;
        opt_total_time += elapsed_ms;
    }

    timing_data[0][2] = opt_gaus_time / num_imgs;
    timing_data[1][2] = opt_dog_time / num_imgs;
    timing_data[2][2] = opt_kps_time / num_imgs;
    timing_data[3][2] = opt_grad_time / num_imgs;
    timing_data[4][2] = opt_desc_time / num_imgs;
    timing_data[5][2] = opt_total_time / num_imgs;


    return timing_data;
}

} // namespace sift
