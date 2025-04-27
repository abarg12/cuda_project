
#ifndef SIFT_H
#define SIFT_H

#include <vector>
#include <array>
#include <cstdint>

#include "image.hpp"

namespace sift {

struct ScaleSpacePyramid {
    int num_octaves;
    int imgs_per_octave;
    std::vector<std::vector<Image>> octaves; 
};

struct Keypoint {
    // discrete coordinates
    int i;
    int j;
    int octave;
    int scale; //index of gaussian image inside the octave

    // continuous coordinates (interpolated)
    float x;
    float y;
    float sigma;
    float extremum_val; //value of interpolated DoG extremum
    
    std::array<uint8_t, 128> descriptor;
};

//*******************************************
// SIFT algorithm parameters, used by default
//*******************************************

// digital scale space configuration and keypoint detection
const int MAX_REFINEMENT_ITERS = 5;
const float SIGMA_MIN = 0.8;
const float MIN_PIX_DIST = 0.5;
const float SIGMA_IN = 0.5;
const int N_OCT = 8;
const int N_SPO = 3;
const float C_DOG = 0.015;
const float C_EDGE = 10;

// computation of the SIFT descriptor
const int N_BINS = 36;
const float LAMBDA_ORI = 1.5;
const int N_HIST = 4;
const int N_ORI = 8;
const float LAMBDA_DESC = 6;

// New constants to support parallel workload
// reduces rounding errors between serial and parallel code
const float M_PIf = 3.14159265358979323846f;
// used as the shared memory size in optimized descriptor kernel
const int DESC_HIST_SHARED_SIZE = (N_HIST * N_HIST * N_ORI);

// feature matching
const float THRESH_ABSOLUTE = 350;
const float THRESH_RELATIVE = 0.7;


/* Summary:
 *     Generates the Guassian scale space pyramid
 * Parameters:
 *   - img: the input image
 *   - sigma_min: base scale for the first image in first octave
 *   - num_octaves: desired number of octaves in the pyramid
 *   - scales_per_octave: number of intervals for scale step
 * Return:
 *     Gaussian blurred images organized by octave and scale
 */
ScaleSpacePyramid generate_gaussian_pyramid(
        const Image& img,
        float sigma_min=SIGMA_MIN,
        int num_octaves=N_OCT,
        int scales_per_octave=N_SPO);


/* Summary:
 *     Generates the Difference of Guassians scale space pyramid
 * Parameters:
 *   - img_pyramid: the input Gaussian scale space pyramid
 * Return:
 *     Structure containing the DoG images at all octaves and scales
 */
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid);


/* Summary:
 *     Identifies the pixel locations of keypoints within the DoG scale space
 * Parameters:
 *   - dog_pyramid: Difference of Gaussians structure, all scales and octaves
 *   - contrast_thresh: threshold used to filter out low-contrast image regions
 *   - edge_thresh: threshold used to filter out keypoints located on edges
 * Return:
 *     Vector of keypoints identified in the image
 */
std::vector<Keypoint> find_keypoints(
        const ScaleSpacePyramid& dog_pyramid,
        float contrast_thresh=C_DOG,
        float edge_thresh=C_EDGE);


/* Summary:
 *     Identifies the pixel locations of keypoints within the DoG pyramid 
 *     scale space. Calls a naive CUDA kernel that works on the pixel-level
 *     granularity to identify extrema which are keypoint candidates.
 * Parameters:
 *   - dog_pyramid: Difference of Gaussians structure, all scales and octaves
 *   - contrast_thresh: threshold used to filter out low-contrast image regions
 *   - edge_thresh: threshold used to filter out keypoints located on edges
 * Return:
 *     Vector of keypoints identified in the image
 */
std::vector<Keypoint> find_keypoints_parallel_naive(
        const ScaleSpacePyramid& dog_pyramid,
        float contrast_thresh=C_DOG,
        float edge_thresh=C_EDGE);


/* Summary:
 *     Calculates the image gradients for each image in the Gaussian pyramid
 * Parameters:
 *   - pyramid: the input Gaussian pyramid
 * Return:
 *     Structure containing the computed gradient images
 */
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid);


/* Summary:
 *     Identifies the dominant orientations for a given keypoint. The orientations are
 *     calculated using the gradients in the neighborhood of the keypoint.
 * Parameters:
 *   - kp: the keypoint object
 *   - grad_pyramid: scale space structure holding the pre-computed image gradients
 *   - lambda_ori: controls the size of the circular patch for keypoint orientations
 *   - lambda_desc: parameter used in the thresholding of keypoints at the image edge
 * Return:
 *     Vector of dominant orientations for the keypoint kp
 */
std::vector<float> find_keypoint_orientations(
        Keypoint& kp,
        const ScaleSpacePyramid& grad_pyramid,
        float lambda_ori=LAMBDA_ORI,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Identifies the dominant orientations for a vector of keypoints.
 *     The orientations are calculated using the gradients in a neighborhood
 *     around each keypoint.
 * Parameters:
 *   - kps: vector
 *   - grad_pyramid: scale space holding the pre-computed image gradients
 *   - lambda_ori: controls the window size used for orientation histogram
 *   - lambda_desc: used in the thresholding of keypoints at the image edge
 * Return:
 *     Vector of dominant orientations for the keypoint kp
 */
std::vector<std::vector<float>> find_keypoint_orientations_parallel_naive(
        std::vector<Keypoint>& kps,
        const ScaleSpacePyramid& grad_pyramid,
        float lambda_ori=LAMBDA_ORI,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Computes the 128-element SIFT descriptor for a single keypoint
 * Parameters:
 *   - kp: keypoint for which the descriptor will be calculated
 *   - theta: the dominant orientation for the keypoint
 *   - grad_pyramid: scale space structure with pre-computed image gradients
 *   - lambda_desc: parameter used in weighting orientations of the descriptor
 * Return:
 *     (void) places the descriptor inside the kp object by reference
 */
void compute_keypoint_descriptor(
        Keypoint& kp,
        float theta,
        const ScaleSpacePyramid& grad_pyramid,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Computes the 128-element SIFT descriptor for all keypoints
 * Parameters:
 *   - kps: keypoints for which the descriptors will be calculated
 *   - theta: the dominant orientation for each keypoint ordered by kps index
 *   - grad_pyramid: scale space holding the pre-computed image gradients
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     (void) places the descriptors inside the kps objects by reference
 */
void compute_keypoint_descriptors_parallel_naive(
        std::vector<Keypoint>& kps,
        std::vector<float> thetas,
        const ScaleSpacePyramid& grad_pyramid,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Computes the orientations and 128-element SIFT descriptors for
 *     all keypoints. Calls a combined orientation and descriptor kernel.
 * Parameters:
 *   - kps: keypoints for which the descriptors will be calculated
 *   - theta: the dominant orientation for each keypoint ordered by kps index
 *   - grad_pyramid: scale space holding the pre-computed image gradients
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     (void) places the descriptors inside the kps objects by reference
 * Optimization notes:
 *     We expected combining the kernels to have an immediate benefit,
 *     but actually the opposite happened since there was too much
 *     pressure on the registers within the complex kernel.
 */
std::vector<Keypoint> find_ori_desc_parallel_combined(std::vector<Keypoint>& kps,
                                                        const ScaleSpacePyramid& grad_pyramid,
                                                        float lambda_ori=LAMBDA_ORI,
                                                        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Computes the orientations and 128-element SIFT descriptors for
 *     all keypoints. Calls the orientation kernel followed by an
 *     optimized descriptor kernel in sequence.
 * Parameters:
 *   - kps: keypoints for which the descriptors will be calculated
 *   - theta: the dominant orientation for each keypoint ordered by kps index
 *   - grad_pyramid: scale space holding the pre-computed image gradients
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     (void) places the descriptors inside the kps objects by reference
 * Optimization notes:
 *     By calling both the orientation and descriptor kernels in the same
 *     host function, we were able to easily share global input arrays for
 *     both kernels. It also gave us the flexibility to focus on the
 *     descriptor generation kernel which was the runtime bottleneck.
 */
std::vector<Keypoint> find_ori_desc_parallel_opt(std::vector<Keypoint>& kps,
                                                        const ScaleSpacePyramid& grad_pyramid,
                                                        float lambda_ori=LAMBDA_ORI,
                                                        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Computes the orientations and SIFT descriptors for all keypoints.
 *     Calls both the orientation generation and descriptor generation kernels.
 * Parameters:
 *   - kps: keypoints for which orientations and descriptors will be calculated
 *   - theta: the dominant orientation for each keypoint ordered by kps index
 *   - grad_pyramid: scale space holding the pre-computed image gradients
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     Vector of keypoints with computed descriptor fields
 */
std::vector<Keypoint> find_ori_desc_parallel_optimized(
        std::vector<Keypoint>& kps,
        const ScaleSpacePyramid& grad_pyramid,
        float lambda_ori=LAMBDA_ORI,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     The SIFT 'main' function which calls all phases. Given an image, will
 *     produce scale-invariant features.
 * Parameters:
 *   - img: the input image which will be scanned for features
 *   - sigma_min: base scale for image in first octave and first sclae
 *   - num_octaves: total number of octaves to generate in scale space
 *   - scales_per_octave: number of scale steps within in octave
 *   - contrast_thresh: threshold used to filter out low contrast keypoints
 *   - edge_thresh: threshold to filter out keypoints located on edges
 *   - lambda_ori: controls the window size used for orientation histogram
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     Vector of keypoint objects representing detected features. Each
 *     keypoint will contain an image location and descriptor.
 */
std::vector<Keypoint> find_keypoints_and_descriptors(
        const Image& img,
        float sigma_min=SIGMA_MIN,
        int num_octaves=N_OCT, 
        int scales_per_octave=N_SPO, 
        float contrast_thresh=C_DOG,
        float edge_thresh=C_EDGE,
        float lambda_ori=LAMBDA_ORI,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     The SIFT 'main' function which calls all phases. Given an image, will
 *     produce scale-invariant features. Calls naive CUDA kernels that exploit
 *     basic data-level parallelism.
 * Parameters:
 *   - img: the input image which will be scanned for features
 *   - sigma_min: base scale for image in first octave and first sclae
 *   - num_octaves: total number of octaves to generate in scale space
 *   - scales_per_octave: number of scale steps within in octave
 *   - contrast_thresh: threshold used to filter out low contrast keypoints
 *   - edge_thresh: threshold to filter out keypoints located on edges
 *   - lambda_ori: controls the window size used for orientation histogram
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     Vector of keypoint objects representing detected features. Each
 *     keypoint will contain an image location and descriptor.
 */
std::vector<Keypoint> find_keypoints_and_descriptors_parallel_naive(
        const Image& img,
        float sigma_min=SIGMA_MIN,
        int num_octaves=N_OCT, 
        int scales_per_octave=N_SPO, 
        float contrast_thresh=C_DOG,
        float edge_thresh=C_EDGE,
        float lambda_ori=LAMBDA_ORI,
        float lambda_desc=LAMBDA_DESC);

/* Summary:
 *     The SIFT 'main' function which calls all phases. Given an image, will
 *     produce scale-invariant features. Calls optimized CUDA kernels that
 *     exploit more thoughtful optimization techniques for parallel workloads.
 * Parameters:
 *   - img: the input image which will be scanned for features
 *   - sigma_min: base scale for image in first octave and first sclae
 *   - num_octaves: total number of octaves to generate in scale space
 *   - scales_per_octave: number of scale steps within in octave
 *   - contrast_thresh: threshold used to filter out low contrast keypoints
 *   - edge_thresh: threshold to filter out keypoints located on edges
 *   - lambda_ori: controls the window size used for orientation histogram
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     Vector of keypoint objects representing detected features. Each
 *     keypoint will contain an image location and descriptor.
 */
std::vector<Keypoint> find_keypoints_and_descriptors_parallel(
        const Image& img,
        float sigma_min=SIGMA_MIN,
        int num_octaves=N_OCT, 
        int scales_per_octave=N_SPO, 
        float contrast_thresh=C_DOG,
        float edge_thresh=C_EDGE,
        float lambda_ori=LAMBDA_ORI,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Timing utility code to find runtime for serial, naive parallel,
 *     and optimized parallel SIFT over a variety of images.
 * Parameters:
 *   - img: the input image which will be scanned for features
 *   - sigma_min: base scale for image in first octave and first sclae
 *   - num_octaves: total number of octaves to generate in scale space
 *   - scales_per_octave: number of scale steps within in octave
 *   - contrast_thresh: threshold used to filter out low contrast keypoints
 *   - edge_thresh: threshold to filter out keypoints located on edges
 *   - lambda_ori: controls the window size used for orientation histogram
 *   - lambda_desc: used in weighting the orientations of the descriptor
 * Return:
 *     Vector of floats representing timing data. 
 */
std::vector<std::vector<float>> find_keypoints_and_descriptors_timing(
        std::vector<Image> imgs,
        float sigma_min=SIGMA_MIN,
        int num_octaves=N_OCT, 
        int scales_per_octave=N_SPO, 
        float contrast_thresh=C_DOG,
        float edge_thresh=C_EDGE,
        float lambda_ori=LAMBDA_ORI,
        float lambda_desc=LAMBDA_DESC);


/* Summary:
 *     Finds matches between two sets of keypoints based on SIFT descriptors
 * Parameters:
 *   - first vector of keypoint objects with populated descriptor fields
 *   - second vector of keypoint objects with populated descriptor fields
 *   - thresh_relative: threshold to determine if descriptors match
 *   - thresh_absolute: threshold for descriptor matching based on
 *                      Euclidean distance between values in descriptor
 * Return:
 *     List of integer pairs (i, j) which represent keypoint matches by index
 */
std::vector<std::pair<int, int>> find_keypoint_matches(
        std::vector<Keypoint>& a,
        std::vector<Keypoint>& b,
        float thresh_relative=THRESH_RELATIVE,
        float thresh_absolute=THRESH_ABSOLUTE);


/* Summary:
 *     Draws locations of keypoints onto an image
 * Parameters:
 *   - img: the input image to draw keypoints onto
 *   - kps: the vector of keypoints which will get drawn
 * Return:
 *     Copy of the input image with keypoints drawn
 */
Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps);


/* Summary:
 *     Draws keypoint matches between two images
 * Parameters:
 *   - a: the first iamge
 *   - b: the second image
 *   - kps_a: vector of keypoints from image 'a'
 *   - kps_b: vector of keypoints from image 'b'
 *   - matches: indices of matches between 'kps_a' and 'kps_b'
 * Return:
 *     New image with both input images side-by-side and matches visualized
 */
Image draw_matches(
        const Image& a,
        const Image& b, std::vector<Keypoint>& kps_a,
        std::vector<Keypoint>& kps_b,
        std::vector<std::pair<int, int>> matches);

} // namespace sift
#endif
