/**
 * @brief      Basic addition.
 *
 *             Here we do basic addition of two integers in a kernel for the
 *             GPU. For doing our first computation, we need to allocate and
 *             afterwards free  memory on the GPU. The syntax to do this is very
 *             similar to what we use in C or C++. Then, to be able to do
 *             multiple additions in parallel, we need to define the number of
 *             threads and blocks.
 *
 *             From: http://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf
 */
#include <iostream>
#include <string>
//#include "opencv2/gpu/gpu.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "device_launch_parameters.h" // for linting


//#define N (2048*2048)
//#define M 512

#define TW 50
#define TH 50

__device__ __forceinline__ uchar euclidean_distance(cv::cuda::PtrStepSz<uchar>v1, cv::cuda::PtrStepSz<uchar>v2) {
    return (uchar)sqrtf((v1[0]-v2[0])*(v1[0]-v2[0])
        +(v1[1]-v2[1])*(v1[1]-v2[1])
        +(v1[2]-v2[2])*(v1[2]-v2[2]));
}

__global__ void euclidDistance(cv::cuda::PtrStepSz<uchar>img, cv::cuda::PtrStepSz<uchar>tile, cv::cuda::PtrStepSz<uchar>out) {

    int y = threadIdx.x; // row
    int x = blockIdx.x; // column

    int sum = 0;

    
    for (int i=0;i<TW;i++) {
        for (int j=0;j<TH;j++) {
            int R_diff = abs(img.ptr(y+j)[(x+i)*3+2]-tile.ptr(j)[i*3+2]);
            int G_diff = abs(img.ptr(y+j)[(x+i)*3+1]-tile.ptr(j)[i*3+1]);
            int B_diff = abs(img.ptr(y+j)[(x+i)*3]-tile.ptr(j)[i*3]);

            sum += (-1*(sqrtf(R_diff*R_diff+G_diff*G_diff+B_diff*B_diff))+442)/(TW*TH);

        }
    }
    

    out.ptr(y)[x*3+2] = 0;//img.ptr(y)[x*3+2];//0;
    out.ptr(y)[x*3+1] = sum/442<=0.5 ? 255 : 255-int((sum/442)-0.5)*2*255;
    out.ptr(y)[x*3] = 0;//img.ptr(y)[x*3];//128;//sum/442<=0.5 ? int((sum/442)*2*255) : 255;

    __syncthreads();
}

/**
 * @brief      Kernel for basic addition.
 *
 * @param      a     Input integer 1
 * @param      b     Input integer 2
 * @param      c     Output integer
 */
//__global__ void add(int *a, int *b, int *c) {
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    if (i < N) {
//        c[i] = a[i] + b[i];
//    }
//}

int main(int argc, char** argv) {

    // Declare copies on host and device
    //int *host_a, *host_b, *host_c; 
    //int *d_a, *d_b, *d_c;
    //int size = N * sizeof(int);

    int TILE_W = 50; //atoi(argv[2]);
    int TILE_H = 50; //atoi(argv[3]);
    int TILE_X = 50; //atoi(argv[4]);
    int TILE_Y = 50; //atoi(argv[5]);

    // Allocate space for device copies of a, b, c
    //cudaMalloc((void **)&d_a, size);
    //cudaMalloc((void **)&d_b, size);
    //cudaMalloc((void **)&d_c, size);

    std::string imp = cv::samples::findFile(argv[1]);

    cv::Mat img = cv::imread(imp, cv::IMREAD_COLOR);
    cv::Mat newImage(img.rows-TILE_H, img.cols-TILE_W, CV_8UC3, cv::Scalar(0,0,255));
    cv::Mat tile(TILE_W, TILE_H, CV_8UC3, cv::Scalar(0,0,0));

    cv::Size s = newImage.size();
    int N = s.height;
    int M = s.width;

    cv::cuda::GpuMat img_device, tile_device, out_device;

    if(TILE_X+TILE_W>img.rows || TILE_X+TILE_W>img.cols){
        std::cout<< "Invalid position or size!" << std::endl;
        exit(0);
    }

    if (img.empty()) {
        std::cout << "Could not read the image: " << imp << std::endl;
        return 1;
    }

    for (int i=TILE_X;i<TILE_X+TILE_W;i++) {
        for (int j=TILE_Y;j<TILE_Y+TILE_H;j++) {
            tile.at<cv::Vec3b>(i-TILE_X,j-TILE_Y) = img.at<cv::Vec3b>(i,j);
        }
    }

    // Copy inputs to device
    //cudaMemcpy(d_a, host_a, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_b, host_b, size, cudaMemcpyHostToDevice);
    img_device.upload(img);
    tile_device.upload(tile);
    out_device.upload(newImage);

    // Launch add() kernel on GPU
    //euclidDistance<<<(N+M-1)/M, M>>>(d_a, d_b, d_c);
    // M - total number of blocks - s.width
    // N - total number of threads in a block - s.height
    euclidDistance<<<M, N>>>(img_device, tile_device, out_device);

    // Copy result back to host
    //cudaMemcpy(host_c, d_c, size, cudaMemcpyDeviceToHost);
    cv::Mat resultHost(out_device);

    imwrite("tile.png", tile);
    imwrite("result.png", resultHost);

    // Output
    std::cout << "Massively parallel szukanie wzorca." << std::endl;
    /*
    std::cout << "Displaying first 10 out of " << N << " results: " << std::endl;
    for (int i = 0; i < 10; ++i)	{
        std::cout << host_a[i] << + " + " << host_b[i] << " = " << host_c[i] << std::endl;
    }
    */

    // Cleanup
    //free(host_a); free(host_b); free(host_c);
    //cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;

}