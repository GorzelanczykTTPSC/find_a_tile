#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

#define TILE_W 51
#define TILE_H 51
#define TILE_X 50
#define TILE_Y 50

double euclideanDistance(cv::Vec3b& v1, cv::Vec3b& v2) {

    //return 

}

int main() {

    std::string imp = cv::samples::findFile("starry_night.jpg");

    cv::Mat img = cv::imread(imp, cv::IMREAD_COLOR);

    cv::Mat half = img(cv::Rect(0,0,300,300));

    cv::Mat newImage(300, 300, CV_8UC3, cv::Scalar(0,0,0));

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

    double max_sum = 0;
    int found_x = -1;
    int found_y = -1;

    for (int i=0;i<half.rows;i++) { // i = row
        std::cout << "i: "+std::to_string(i) << std::endl;
        for (int j=0;j<half.cols;j++) { // j = col
            
            double sum = 0;

            for (int k=0;k<TILE_W;k++) {
                for (int l=0;l<TILE_H;l++) {

                    double val = std::max((-1*std::max((int)cv::norm(half.at<cv::Vec3b>(i+k,j+l), tile.at<cv::Vec3b>(k, l)), 0)+255), 0);
                    //if (val!=0) std::cout << val << std::endl;
                    sum += val/(TILE_W*TILE_H);

                }
            }

            if (sum>max_sum) {
                max_sum=sum; std::cout<<"Max sum: " << max_sum << std::endl;
                found_x = i;
                found_y = j;
            }

            newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,sum);
        }
    }

    std::cout << "Coords: " << found_x << " " << found_y << std::endl;

    // distance == 0 gdy są identyczne
    //std::cout << "Distance between two random vectors: " << cv::norm(img.at<cv::Vec3b>(60,60), tile.at<cv::Vec3b>(40,10)) << std::endl;

    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;

    //cv::minMaxLoc(newImage, &minVal, &maxVal, &minLoc, &maxLoc );

    std::cout << "Max loc: " << maxLoc << std::endl;

     imshow("Display window", newImage);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night2.png", tile);
    }
    return 0;

}