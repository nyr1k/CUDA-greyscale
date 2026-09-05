#include <iostream>
#include <string>
#include <chrono> 

#include <opencv2/imgcodecs.hpp>

__global__ void greyscale_kernel(const unsigned char* in, 
                          unsigned char* out, 
                          int width, 
                          int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y *  blockIdx.y + threadIdx.y;

    if (x < width && y < height) {
        int grayOffset = y * width + x;
        int rgbOffset = grayOffset * 3;

        out[grayOffset] = 0.21f*in[rgbOffset] + 0.71f*in[rgbOffset+1] + 0.07f*in[rgbOffset+2];
    }
}

__host__ unsigned char* greyscale_cpu(const cv::Mat image,
                            int width,
                            int height)
{
    size_t rgbSize = width * height * 3 * sizeof(unsigned char);
    size_t greySize = width * height * 1 * sizeof(unsigned char);

    unsigned char *image_in = nullptr;
    unsigned char *image_out = nullptr;
    image_in = new unsigned char[rgbSize];
    image_out = new unsigned char[greySize];

    std::memcpy(image_in, image.data, rgbSize);

    for (int i = 0; i < greySize; i++) {
        int rgbOffset = i * 3;
        image_out[i] = 0.21f*image_in[rgbOffset] + 0.71f*image_in[rgbOffset+1] + 0.07f*image_in[rgbOffset+2];
    }

    delete[] image_in;
    return image_out;
}

unsigned char* greyscale_gpu(const cv::Mat image, int width, int height) 
{
    
    size_t rgbSize = width * height * 3 * sizeof(unsigned char);
    size_t greySize = width * height * 1 * sizeof(unsigned char);

    unsigned char *image_in = nullptr;
    unsigned char *image_out = nullptr;
    cudaMallocManaged((void**) &image_in, rgbSize);
    cudaMallocManaged((void **) &image_out, greySize);

    std::memcpy(image_in, image.data, rgbSize);
    
    dim3 dimgrid(std::ceil(width/16.0), std::ceil(height/16.0));
    dim3 dimBlock(16, 16, 1); 

    greyscale_kernel<<<dimgrid, dimBlock>>>(image_in, image_out, width, height);
    cudaDeviceSynchronize();

    cudaFree(image_in);

    return image_out;
}

int main(int argc, char* argv[]) 
{
    
    if (argc == 1) {
        std::cerr << "No input provided" << std::endl;
        return 1;
    }

    if (argc != 3) {
        std::cerr << "Choose execution device: cpu/gpu" << std::endl;
        return 1;
    }

    const std::string image_path(argv[1]);
    const std::string execution(argv[2]);
    
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_COLOR_RGB);
    if (image.empty()) {
        std::cerr << "Could not read the provided image" << std::endl;
        return 1;
    }

    int width = image.cols, height = image.rows;

    unsigned char* image_out = nullptr;
    auto start = std::chrono::high_resolution_clock::now();
    if (execution == "gpu") 
        image_out = greyscale_gpu(image, width, height);
    else if (execution == "cpu")
        image_out = greyscale_cpu(image, width, height);
    else {
        std::cerr << "Choose execution device: cpu/gpu" << std::endl;
        return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Gray scale function execution time: " << duration.count() << " milliseconds" << std::endl;

    cv::Mat gray_image(height, width, CV_8UC1, image_out);

    if (!cv::imwrite("gray.jpeg", gray_image)) {
        std::cerr << "Image write failed" << std::endl;
        return 1;
    }

    if (execution == "gpu")
        cudaFree(image_out);
    else if (execution == "cpu")
        delete[] image_out;

    return 0;
}