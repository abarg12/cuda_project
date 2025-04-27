#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 1) {
        std::cerr << "Usage: ./time_all\n";
        return 0;
    }

    Image img0("../imgs/ParallelTestData/boat_harbor.jpg");
    Image img1("../imgs/ParallelTestData/cat.jpg");
    Image img2("../imgs/ParallelTestData/cat2.jpg");
    Image img3("../imgs/ParallelTestData/cat3.jpg");
    Image img4("../imgs/ParallelTestData/city.jpg");
    Image img5("../imgs/ParallelTestData/construction_truck.jpg");
    Image img6("../imgs/ParallelTestData/kid_homework.jpg");
    Image img7("../imgs/ParallelTestData/motorcycles_city.jpg");
    Image img8("../imgs/ParallelTestData/multiple_tvs.jpg");
    Image img9("../imgs/ParallelTestData/woman_with_telephone.jpg");

    std::vector<Image> imgs = {
        img0, img1, img2, img3, img4,
        img5, img6, img7, img8, img9
    };

    for (auto& img : imgs) {
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
    }

    printf("running time tests on 10 different images...\n\n");

    std::vector<std::vector<float>> time_data = sift::find_keypoints_and_descriptors_timing(imgs);

    printf("SERIAL TIMING ----------------------------------------------\n");
    printf("Gaussian Pyramid: %.3f ms\n", time_data[0][0]);
    printf("DoG Pyramid:      %.3f ms\n", time_data[1][0]);
    printf("Find Keypoints:   %.3f ms\n", time_data[2][0]);
    printf("Gradient Pyramid: %.3f ms\n", time_data[3][0]);
    printf("Descriptors:      %.3f ms\n", time_data[4][0]);
    printf("Total SIFT Time:  %.3f ms\n", time_data[5][0]);
    printf("\n");

    printf("NAIVE PARALLEL TIMING --------------------------------------\n");
    printf("Gaussian Pyramid: %.3f ms\n", time_data[0][1]);
    printf("DoG Pyramid:      %.3f ms\n", time_data[1][1]);
    printf("Find Keypoints:   %.3f ms\n", time_data[2][1]);
    printf("Gradient Pyramid: -\n", time_data[3][1]);
    printf("Descriptors:      %.3f ms\n", time_data[4][1]);
    printf("Total SIFT Time:  %.3f ms\n", time_data[5][1]);
    printf("\n");

    printf("OPTIMIZED PARALLEL TIMING ----------------------------------\n");
    printf("Gaussian Pyramid: %.3f ms\n", time_data[0][2]);
    printf("DoG Pyramid:      %.3f ms\n", time_data[1][2]);
    printf("Find Keypoints:   -\n", time_data[2][2]);
    printf("Gradient Pyramid: -\n", time_data[3][2]);
    printf("Descriptors:      %.3f ms\n", time_data[4][2]);
    printf("Total SIFT Time:  %.3f ms\n", time_data[5][2]);
    printf("\n");

    return 0;
}