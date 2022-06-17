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

__global__ void euclidDistance(cv::cuda::PtrStepSz<uchar3>img, cv::cuda::PtrStepSz<uchar3>tile, cv::cuda::PtrStepSz<uchar3>out) {

    //int y = threadIdx.x; // row
    //int x = blockIdx.x; // column
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0;
    
    for (int i=0;i<TW;i++) {
        for (int j=0;j<TH;j++) {
            uchar3 pix = img(row+j, col+i);
            uchar3 tilepix = tile(j, i);
            float R_diff = (float)(pix.z-tilepix.z);
            float G_diff = (float)(pix.y-tilepix.y);
            float B_diff = (float)(pix.x-tilepix.x);

            sum += (-1*(sqrtf(R_diff*R_diff+G_diff*G_diff+B_diff*B_diff))+442)/(TW*TH); // R_diff+G_diff+B_diff<5?1:0;

        }
    }
    
    out(row, col).x = 0;//img.ptr(y)[x*3+2];//0;
    out(row, col).y = sum/442.0<=0.5 ? 255 : 255-int((sum/442.0)-0.5)*2*255;
    out(row, col).z = sum/442.0<=0.5 ? int((sum/442.0)*2*255) : 255; // sum/10;

    //__syncthreads();
}

int main(int argc, char** argv) {

    if (argc<6){
        std::cout << "Usage: " << argv[0] << " <image> <fragment width> <fragmend height> <X fragment position> <Y fragmend position> [testrun]" <<std::endl;
        exit(1);
    }

    int TILE_W = atoi(argv[2]);
    int TILE_H = atoi(argv[3]);
    int TILE_X = atoi(argv[4]);
    int TILE_Y = atoi(argv[5]);
    bool testrun = false;

    if (argc>6) {
        testrun = atoi(argv[6])==1?true:false;
    }


    // Allocate space for device copies of a, b, c
    //cudaMalloc((void **)&d_a, size);
    //cudaMalloc((void **)&d_b, size);
    //cudaMalloc((void **)&d_c, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;

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

    cudaEventRecord( start, 0 );

    // Launch add() kernel on GPU
    //euclidDistance<<<(N+M-1)/M, M>>>(d_a, d_b, d_c);
    // M - total number of blocks - s.width
    // N - total number of threads in a block - s.height
    // M, N
    dim3 Threads(32, 16);
    dim3 Blocks((newImage.cols + Threads.x - 1)/Threads.x, (newImage.rows + Threads.y - 1)/Threads.y);
    euclidDistance<<<Blocks, Threads>>>(img_device, tile_device, out_device);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    if (testrun){
        std::cout<< time;
    }
    else{
    std::cout << "Exec time: " << time << " ms" << std::endl;
    }

    // Copy result back to host
    //cudaMemcpy(host_c, d_c, size, cudaMemcpyDeviceToHost);
    cv::Mat resultHost(out_device);

    imwrite("tile.png", tile);
    imwrite("result.png", resultHost);



    return 0;

}