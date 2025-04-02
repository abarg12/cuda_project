#include "image.hpp"
#include "sift.hpp"

#define ENABLE_TIMING

int main(int argc, char *argv[])
{
    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(img);

    std::cout << "Found " << kps.size() << " keypoints. Output image is saved as result.jpg\n";
    return 0;
}