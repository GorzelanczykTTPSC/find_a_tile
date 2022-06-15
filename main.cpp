#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

#define TILE_W 51
#define TILE_H 51
#define TILE_X 50
#define TILE_Y 50

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

    for (int i=0;i<half.rows;i++) {
        for (int j=0;j<half.cols;j++) {
            newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,-1*std::max((int)cv::norm(half.at<cv::Vec3b>(i,j), tile.at<cv::Vec3b>(i, j)), 255)+255);
        }
    }

    // distance == 0 gdy są identyczne
    std::cout << "Distance between two random vectors: " << cv::norm(img.at<cv::Vec3b>(60,60), tile.at<cv::Vec3b>(40,10)) << std::endl;

     imshow("Display window", newImage);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night2.png", tile);
    }
    return 0;

}