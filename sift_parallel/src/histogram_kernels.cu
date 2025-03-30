// __global__ void histogram_kernel() {
//     return;
// }

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
            // float val = img.get_pixel(x, y, 0);
            float val = image[image_idx];
            float neighbor = 0;
            int offset = 0;

            // for (int dx : {-1,0,1}) {
                // for (int dy : {-1,0,1}) {
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    // neighbor = prev.get_pixel(x+dx, y+dy, 0);
                    offset = (dy * image_width) + dx;
                    if (image_idx + offset < 0 || image_idx + offset >= image_size) {
                        continue;
                    }

                    neighbor = image[image_down_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    // neighbor = next.get_pixel(x+dx, y+dy, 0);
                    neighbor = image[image_up_idx + (dy * image_width) + dx];
                    if (neighbor > val) is_max = false;
                    if (neighbor < val) is_min = false;

                    // neighbor = img.get_pixel(x+dx, y+dy, 0);
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