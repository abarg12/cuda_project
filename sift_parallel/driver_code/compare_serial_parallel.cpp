#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 2) {
        std::cerr << "Usage: ./compare_serial_parallel input.jpg (or .png)\n";
        return 0;
    }
    Image img(argv[1]);
    img =  img.channels == 1 ? img : rgb_to_grayscale(img);

    // Runs the serial version of SIFT
    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(img);

    // Runs the unoptimized parallel version of SIFT
    std::vector<sift::Keypoint> kps_p = sift::find_keypoints_and_descriptors_parallel_naive(img);

    // Runs the optimized parallel version of SIFT
    std::vector<sift::Keypoint> kps_p_opt = sift::find_keypoints_and_descriptors_parallel(img);

    // Verifying keypoints for accuracy among all versions
    printf("--------------------------------------------------\n");
    printf("VERIFYING ACCURACY OF PARALLEL VERSIONS\n");
    printf("Serial: %d keypoints, Naive Parallel: %d keypoints\n", kps.size(), kps_p.size());
    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps, kps_p);
    printf("Serial and Naive Parallel have %d matching keypoints\n", matches.size());

    printf("Serial: %d keypoints, Optimized Parallel: %d keypoints\n", kps.size(), kps_p_opt.size());
    std::vector<std::pair<int, int>> opt_matches = sift::find_keypoint_matches(kps, kps_p_opt);
    printf("Serial and Optimized Parallel have %d matching keypoints\n", opt_matches.size());

    return 0;
}