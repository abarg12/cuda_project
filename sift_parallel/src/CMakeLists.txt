add_library(stb_image STATIC
            stb_image.h
            stb_image_write.h
)

set_target_properties(stb_image PROPERTIES LINKER_LANGUAGE CXX)

add_library(img STATIC
            image.hpp
            image.cpp
)

target_link_libraries(img PRIVATE stb_image)

add_library(sift STATIC
            sift.cu
            sift.hpp
            gaussian_kernels.cu
            histogram_kernels.cu
)

target_link_libraries(sift PRIVATE
                      img
)
