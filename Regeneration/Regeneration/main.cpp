#include <opencv.hpp>
#include <cv.h>

using namespace cv;

Mat composit(Mat srcA, Mat srcB, Mat srcC) {
	Mat res = srcB;
	for (int i = 0; i < srcB.rows; i++) {
		for (int j = 0; j < srcC.cols; j++) {
			if (srcC.at<int>(i,j) != 0 && srcB.at<int>(i,j) == 0) {
				res.at<int>(i,j) = srcC.at<int>(i,j);
			}
		}
	}
	return res;
}

int main() {
	Mat srcA = imread("data/A.png");
	Mat srcB = imread("data/B.png");
	Mat srcC = imread("data/C.png");
	imshow("A", srcA);
	imshow("B", srcB);
	imshow("C", srcC);
	waitKey(0);
}