#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 2) {
        std::cerr << "Usage: ./find_keypoints input.jpg (or .png)\n";
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
    if (kps.size() != kps_p.size()) {
        printf("[!!!FAILURE!!!] Serial: %d keypoints, Naive Parallel: %d keypoints\n", kps.size(), kps_p.size());
    } else {
        printf("[SUCCESS] Serial: %d keypoints, Naive Parallel: %d keypoints\n", kps.size(), kps_p.size());

        bool same_locations = true;
        for (int i = 0; i < kps.size(); i++) {
            if (kps[i].i != kps_p[i].i) same_locations = false;
            if (kps[i].j != kps_p[i].j) same_locations = false;
        }
        if (same_locations) {
            printf("[SUCCESS] Serial and Naive Parallel keypoint locations match\n");

            bool same_descriptors = true;
            for (int i = 0; i < kps.size(); i++) {
                for (int j = 0; j < 128; j++) {
                    // allow for descriptor to be +/-10 due to rounding differences on GPU and CPU
                    if (std::abs(kps[i].descriptor[j] - kps_p[i].descriptor[j]) > 10) {
                        same_descriptors = false;
                    }
                }
            }
            if (same_descriptors) {
                printf("[SUCCESS] Serial and Naive Parallel descriptors match\n");
            } else {
                 printf("[!!!FAILURE!!!] Serial and Naive Parallel descriptors do not match\n");
            }

        } else {
            printf("[!!!FAILURE!!!] Serial and Naive Parallel keypoint locations do not match\n");
        }
    }

    if (kps.size() != kps_p_opt.size()) {
        printf("[!!!FAILURE!!!] Serial: %d keypoints, Optimized Parallel: %d keypoints\n", kps.size(), kps_p_opt.size());
    } else {
        printf("[SUCCESS] Serial: %d keypoints, Optimized Parallel: %d keypoints\n", kps.size(), kps_p_opt.size());
        bool same_locations = true;
        for (int i = 0; i < kps.size(); i++) {
            if (kps[i].i != kps_p_opt[i].i) same_locations = false;
            if (kps[i].j != kps_p_opt[i].j) same_locations = false;
        }
        if (same_locations) {
            printf("[SUCCESS] Serial and Optimized Parallel keypoint locations match\n");

            bool same_descriptors = true;
            for (int i = 0; i < kps.size(); i++) {
                for (int j = 0; j < 128; j++) {
                    // allow for descriptor to be +/-10 due to rounding differences on GPU and CPU
                    if (std::abs(kps[i].descriptor[j] - kps_p_opt[i].descriptor[j]) > 10) {
                        same_descriptors = false;
                    }
                }
            }
            if (same_descriptors) {
                printf("[SUCCESS] Serial and Optimized Parallel descriptors match\n");
            } else {
                 printf("[!!!FAILURE!!!] Serial and Optimized Parallel descriptors do not match\n");
            }

        } else {
            printf("[!!!FAILURE!!!] Serial and Optimized Parallel keypoint locations do not match\n");
        }
    }

    return 0;
}