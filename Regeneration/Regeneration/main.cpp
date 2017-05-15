#include <opencv.hpp>
#include <cv.h>

using namespace cv;

int main() {
	Mat src = imread("data/B.png");
	imshow("blackwhite", src);
	waitKey(0);
}