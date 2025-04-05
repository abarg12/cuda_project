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

    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(img);
    printf("[Serial] Found %d keypoints\n", kps.size());

    std::vector<sift::Keypoint> kps_p = sift::find_keypoints_and_descriptors_parallel(img);
    printf("[Parallel] Found %d keypoints\n", kps_p.size());

    return 0;
}