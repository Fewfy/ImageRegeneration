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
	Mat srcB = imread("data/B.png");
	imshow("origin", srcB);
	float a = 0.5;
	float lamda = 0.2;
	for (int t = 0; t < 300; t++) {
		for (int i = 1; i < srcB.rows-1; i++) {
			for (int j = 1; j < srcB.cols-1; j++) {
				//ÐèÒªÐÞ¸´
				if (srcB.ptr<float>(i)[j] == 0.0 || srcB.ptr<float>(i)[j] == -0.0) {
					float Uo = srcB.ptr<float>(i)[j];
					float Un = srcB.ptr<float>(i - 1)[j];
					float Ue = srcB.ptr<float>(i)[j + 1];
					float Us = srcB.ptr<float>(i + 1)[j];
					float Uw = srcB.ptr<float>(i)[j - 1];

					float UNE = srcB.ptr<float>(i - 1)[j + 1];
					float UNW = srcB.ptr<float>(i - 1)[j - 1];
					float USE = srcB.ptr<float>(i + 1)[j + 1];
					float USW = srcB.ptr<float>(i + 1)[j - 1];
				}
			}
		}
	}


	imshow("B", srcB);
	waitKey(0);
}