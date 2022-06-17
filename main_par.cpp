#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <cstdlib>




double euclideanDistance(cv::Vec3b& v1, cv::Vec3b& v2) {

    return sqrt((v1[0]-v2[0])*(v1[0]-v2[0])
        +(v1[1]-v2[1])*(v1[1]-v2[1])
        +(v1[2]-v2[2])*(v1[2]-v2[2]));

}

std::vector<int> gradient(float value){
    std::vector<int> out;
    if (value<=0.5){
       
        out.push_back(0);
        out.push_back(255);
        out.push_back(int(value*2*255));
        return out;
    }
    else {
        out.push_back(0);
        out.push_back(255 - int((value-0.5)*2*255));
        out.push_back(255);
       
        return out;
    }
}
    

int main(int argc, char *argv[]) {

    
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
        testrun = atoi(argv[6])==1;
    }
    
    std::string imp = cv::samples::findFile(argv[1]);
    cv::Mat img = cv::imread(imp, cv::IMREAD_COLOR);

    if(TILE_X+TILE_W>img.rows || TILE_X+TILE_W>img.cols){
        std::cout<< "Invalid position or size!" << std::endl;
        exit(0);
    }

    cv::Mat newImage(img.rows-TILE_H, img.cols-TILE_W, CV_8UC3, cv::Scalar(0,0,0));

    if (img.empty()) {
        std::cout << "Could not read the image: " << imp << std::endl;
        return 1;
    }

    cv::Mat tile(TILE_W, TILE_H, CV_8UC3, cv::Scalar(0,0,0));

    for (int i=TILE_X;i<TILE_X+TILE_W;i++) {
        for (int j=TILE_Y;j<TILE_Y+TILE_H;j++) {
            tile.at<cv::Vec3b>(i-TILE_X,j-TILE_Y) = img.at<cv::Vec3b>(i,j);
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    double max_sum = 0;
    int found_x = -1;
    int found_y = -1;

    #pragma omp parallel for shared(found_x, found_y)
    for (int i=0;i<img.rows-TILE_W;i++) { // i = row
        // std::cout << "i: "+std::to_string(i) << std::endl;
        #pragma omp parallel for shared(found_x, found_y)
        for (int j=0;j<img.cols-TILE_H;j++) { // j = col
            
            double sum = 0;

            #pragma omp parallel for reduction(+:sum)
            for (int k=0;k<TILE_W;k++) {
                for (int l=0;l<TILE_H;l++) {

                    double val = -1*(int)euclideanDistance(img.at<cv::Vec3b>(i+k,j+l), tile.at<cv::Vec3b>(k, l))+442;
                    sum += val/(TILE_W*TILE_H);

                }
            }

            if (sum>max_sum) {
                max_sum=sum;
                found_x = i;
                found_y = j;
            }

            std::vector<int> color = gradient(sum/442);
            newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(color[0],color[1],color[2]);
            // newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,sum);

        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    if (testrun) {
        std::cout << (end_time-start_time)/std::chrono::milliseconds(1);
        return 0;
    }

    std::cout << "Time: " << (end_time-start_time)/std::chrono::milliseconds(1) << " ms" << std::endl;
    std::cout << "Coords: " << found_x << " " << found_y << std::endl;


    cv::rectangle(newImage, cv::Point(found_x, found_y), cv::Point(found_x+TILE_W, found_y+TILE_H), cv::Scalar(255,0,0));
    cv::rectangle(newImage, cv::Point(found_x-1, found_y-2), cv::Point(found_x+TILE_W+1, found_y+TILE_H+1), cv::Scalar(255,255,255));

    cv::rectangle(img, cv::Point(found_x, found_y), cv::Point(found_x+TILE_W, found_y+TILE_H), cv::Scalar(255,0,0));
    cv::rectangle(img, cv::Point(found_x-1, found_y-2), cv::Point(found_x+TILE_W+1, found_y+TILE_H+1), cv::Scalar(255,255,255));

    // distance == 0 gdy są identyczne
    std::cout << "Distance between two random vectors: " << cv::norm(img.at<cv::Vec3b>(60,60), tile.at<cv::Vec3b>(40,10)) << std::endl;


    imshow("Source image window", img);

    imshow("Display window", newImage);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite(std::string(argv[1])+"_out.png", newImage);
        imwrite(std::string(argv[1])+"_out2.png", img);
    }
    
    return 0;

}