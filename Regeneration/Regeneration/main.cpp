#include <opencv.hpp>
#include <cv.h>

using namespace cv;
#define N 3
bool GetMatrixInverse(double src[N][N], int n, double des[N][N]);
double getA(double arcs[N][N], int n);
void  getAStart(double arcs[N][N], int n, double ans[N][N]);


void InverseMatrix(double A[3][3], int n,double B[3][3])
{
	int i, j, k, m = 2 * n;
	double mik, temp;
	double **a = new double*[n];

	for (i = 0; i<n; i++)
	{
		a[i] = new double[2 * n];
	}

	for (i = 0; i<n; i++)
	{
		for (j = 0; j<n; j++)
		{
			if (i == j)
				B[i][j] = 1.0;
			else
				B[i][j] = 0.0;
		}
	}        //初始化B=E

	for (i = 0; i<n; i++)
		for (j = 0; j<n; j++)
			a[i][j] = A[i][j];  //复制A到a，避免改变A的值
	for (i = 0; i<n; i++)
		for (j = n; j<m; j++)
			a[i][j] = B[i][j - n];  //复制B到a，增广矩阵

	for (k = 1; k <= n - 1; k++)
	{
		for (i = k + 1; i <= n; i++)
		{
			mik = a[i - 1][k - 1] / a[k - 1][k - 1];
			for (j = k + 1; j <= m; j++)
			{
				a[i - 1][j - 1] -= mik*a[k - 1][j - 1];
			}
		}
	}        //顺序高斯消去法化左下角为零

	for (i = 1; i <= n; i++)
	{
		temp = a[i - 1][i - 1];
		for (j = 1; j <= m; j++)
		{
			a[i - 1][j - 1] /= temp;
		}
	}        //归一化

	for (k = n - 1; k >= 1; k--)
	{
		for (i = k; i >= 1; i--)
		{
			mik = a[i - 1][k];
			for (j = k + 1; j <= m; j++)
			{
				a[i - 1][j - 1] -= mik*a[k][j - 1];
			}
		}
	}        //逆序高斯消去法化增广矩阵左边为单位矩阵

	for (i = 0; i<n; i++)
		for (j = 0; j<n; j++)
			B[i][j] = a[i][j + n];  //取出求逆结果

	for (i = 0; i<n; i++)
		for (j = 0; j<n; j++)
			if (fabs(B[i][j])<0.0001)
				B[i][j] = 0.0;

	delete[]a;
}
//得到给定矩阵src的逆矩阵保存到des中。
bool GetMatrixInverse(double src[N][N], int n, double des[N][N])
{
	double flag = getA(src, n);
	double t[N][N];
	if (flag == 0)
	{
		return false;
	}
	else
	{
		getAStart(src, n, t);
		for (int i = 0; i<n; i++)
		{
			for (int j = 0; j<n; j++)
			{
				des[i][j] = t[i][j] / flag;
			}

		}
	}


	return true;

}

//按第一行展开计算|A|
double getA(double arcs[N][N], int n)
{
	if (n == 1)
	{
		return arcs[0][0];
	}
	double ans = 0;
	double temp[N][N] = { 0.0 };
	int i, j, k;
	for (i = 0; i<n; i++)
	{
		for (j = 0; j<n - 1; j++)
		{
			for (k = 0; k<n - 1; k++)
			{
				temp[j][k] = arcs[j + 1][(k >= i) ? k + 1 : k];

			}
		}
		double t = getA(temp, n - 1);
		if (i % 2 == 0)
		{
			ans += arcs[0][i] * t;
		}
		else
		{
			ans -= arcs[0][i] * t;
		}
	}
	return ans;
}

//计算每一行每一列的每个元素所对应的余子式，组成A*
void  getAStart(double arcs[N][N], int n, double ans[N][N])
{
	if (n == 1)
	{
		ans[0][0] = 1;
		return;
	}
	int i, j, k, t;
	double temp[N][N];
	for (i = 0; i<n; i++)
	{
		for (j = 0; j<n; j++)
		{
			for (k = 0; k<n - 1; k++)
			{
				for (t = 0; t<n - 1; t++)
				{
					temp[k][t] = arcs[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
				}
			}


			ans[j][i] = getA(temp, n - 1);
			if ((i + j) % 2 == 1)
			{
				ans[j][i] = -ans[j][i];
			}
		}
	}
}

//假设每个像素和周围24个点每个通道的值符合高斯分布，进行极大似然估计
IplImage* MaximumLikely(IplImage*src) {
	int row;
	int col;
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	double res = 0;
	int t;
	bool flag;
	int sum[3] = { 0,0,0 };
	int pixels[3] = { 0,0,0 };
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] == 0)
					flag = true;
			}
			if (flag) {
				sum[0] = sum[1] = sum[2] = 0;
				pixels[0] = pixels[1] = pixels[2] = 0;
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						CvScalar cs = cvGet2D(src, row, col);
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							for (int k = 0; k < src->nChannels; k++) {
								if (cs.val[k] != 0) {
									sum[k] += cs.val[k];
									pixels[k] ++;
								}
							}
						}
					}
				}
				for (int k = 0; k < src->nChannels; k++) {
					if (cs.val[k] == 0) {
						double miu = sum[k] / pixels[k];
						double row_phi;
						double col_phi;
						for (int m = -2; m <= 2; m++) {
							row = i + m;
							for (int n = -2; n <= 2; n++) {
								col = j + n;
								CvScalar cs = cvGet2D(src, row, col);
								if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
									for (int k = 0; k < src->nChannels; k++) {
										if (cs.val[k] != 0) {
											
										}
									}
								}
							}
						}
					}
				}
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}

Mat composit(Mat srcA, Mat src, Mat srcC) {
	Mat res = src;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < srcC.cols; j++) {
			if (srcC.at<int>(i,j) != 0 && src.at<int>(i,j) == 0) {
				res.at<int>(i,j) = srcC.at<int>(i,j);
			}
		}
	}
	return res;
}

IplImage* GM(double phi,IplImage*src) {
	int row;
	int col;
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	double res = 0;
	int t;
	double weight;
	bool flag;
	double sum[3] = { 0,0,0 };
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] == 0)
					flag = true;
			}
			if (flag) {
				sum[0] = sum[1] = sum[2] = 0;
				for (int m = -3; m <= 3; m++) {
					row = i + m;
					for (int n = -3; n <= 3; n++) {
						col = j + n;
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							CvScalar cv = cvGet2D(src, row, col);
							for (int k = 0; k < src->nChannels; k++) {
								if (cv.val[k] != 0) {
									weight = pow(phi, 2) / pow(pow(phi, 2) + pow(m, 2) + pow(n, 2), 2);
									sum[k] += weight * cv.val[k];
								}
							}
								
						}
					}
				}
				for (int k = 0; k < src->nChannels; k++) {
					if (cs.val[k] == 0) {
						cs.val[k] = sum[k];
					}
				}
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}

int degrade(IplImage* img,int i,int j,int t) {
	double delta = 0.5;
	double lamda = 0.2;
	CvScalar center = cvGet2D(img, i, j);
	CvScalar north = cvGet2D(img, i - 1, j);
	CvScalar east = cvGet2D(img, i, j + 1);
	CvScalar south = cvGet2D(img, i + 1, j);
	CvScalar west = cvGet2D(img, i, j - 1);
	CvScalar noEast = cvGet2D(img, i - 1, j + 1);
	CvScalar eaSouth = cvGet2D(img, i + 1, j + 1);
	CvScalar soWest = cvGet2D(img, i + 1, j - 1);
	CvScalar weNorth = cvGet2D(img, i - 1, j - 1);
	double Un = sqrt(pow(center.val[t] + north.val[t], 2) + pow((noEast.val[t] - weNorth.val[t]) / 2, 2));
	double Ue = sqrt(pow(center.val[t] + east.val[t], 2) + pow((noEast.val[t] - eaSouth.val[t]) / 2, 2));
	double Us = sqrt(pow(center.val[t] + south.val[t], 2) + pow((eaSouth.val[t] - soWest.val[t]) / 2, 2));
	double Uw = sqrt(pow(center.val[t] + west.val[t], 2) + pow((soWest.val[t] - weNorth.val[t]) / 2, 2));

	double Wn = 1.0 / sqrt(Un * Un + delta * delta);
	double We = 1.0 / sqrt(Ue * Ue + delta * delta);
	double Ws = 1.0 / sqrt(Us * Us + delta * delta);
	double Ww = 1.0 / sqrt(Uw * Uw + delta * delta);

	double sum = Wn + We + Ws + Ww + lamda;

	double Hon = Wn / sum;
	double Hoe = We / sum;
	double How = Ww / sum;
	double Hos = Ws / sum;
	double Hoo = lamda / sum;
	return (int)(Hon * north.val[t] + Hoe * east.val[t] + How * west.val[t] + Hos * south.val[t] + Hoo * center.val[t]);
}


IplImage *regression(IplImage*src) {
	int row;
	int col;
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	double res = 0;
	int t;
	bool flag;
	int pixels[3] = { 0,0,0 };
	double sum[3] = { 0,0,0 };
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] == 0)
					flag = true;
			}
			if (flag) {
				sum[0] = sum[1] = sum[2] = 0;
				pixels[0] = pixels[1] = pixels[2] = 0;
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							CvScalar cv = cvGet2D(src, row, col);
							for (int k = 0; k < src->nChannels; k++) {
								if (cv.val[k] != 0) {
									sum[k] += 1.0 /(exp(sqrt(pow(m, 2) + pow(n, 2)))) * cv.val[k];
								}
							}
						}
					}
				}
				for (int k = 0; k < src->nChannels; k++) {
					cs.val[k] = (int)sum[k];
				}
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}

void matrixmultiply(double a[][N],double b[][N], int row,int col,double dst[3][3]) {
	
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dst[i][j] = 0;
			for (int ix = 0; ix < row; ix++) {
				dst[i][j] += a[i][ix] * b[ix][j];
			}
		}
	}
}

void pointmatrixmultiply(double a[][N], double b[][N], int row, int col, double dst[3][3]) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dst[i][j] = a[i][j] * b[i][j];
		}
	}
}

IplImage* LinearRegression(IplImage* src) {
	int row;
	int col;
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	
	int t;
	bool flag;
	int count = 0;
	double a[3][3];
	double b[3][3];
	double res[3][3];
	double beta[3][3];
	double inverse[3][3];

	double inversey[3][3];
	double y[3][3];
	bool out = false;
	bool ok = false;
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] == 0)
					flag = true;
			}
			if (flag) {
				
				count = 0;
				out = false;
				for (t = 0; t < src->nChannels; t++) {
					if (cs.val[t] == 0) {
						ok = false;
						for (int m = -2; m <= 2; m++) {
							
							if (out)
								break;
							for (int n = -2; n <= 2; n++) {
								if (out)
									break;
								row = i + m;
								col = j + n;

								if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
									CvScalar cv = cvGet2D(src, row, col);
									if (cv.val[t] != 0) {

										if (count < 3) {
											
											a[count][0] = 1;
											a[count][1] = row;
											a[count][2] = col;
										}
										else if (count == 3) {
											b[0][0] = 3;
											int sum_x;
											int sum_y;
											sum_x = a[0][1] + a[1][1] + a[2][1];
											sum_y = a[0][2] + a[1][2] + a[2][2];
											b[0][1] = sum_x;
											b[0][2] = sum_y;
											b[1][0] = b[1][1] = b[1][2] = sum_x;
											b[2][0] = b[2][1] = b[2][2] = sum_y;
											//x_ 乘以x得到res矩阵
											matrixmultiply(b, a, 3, 3, res);
											//res矩阵求逆
											if (!GetMatrixInverse(res, 3, inverse)) {
												count = 0;
												m -= 1;
												n -= 1;
												break;
											}
											ok = true;
											//x_和像素值矩阵相乘得到inversey
											matrixmultiply(b, y, 3, 3, inversey);
											//inverse和inversey相乘得到beta
											matrixmultiply(inverse, inversey, 3, src->nChannels, beta);
											out = true;
											break;
										}
										y[count][t] = cv.val[t];
										count++;
									}
									
								}
							}
						}
						if (ok)
							cs.val[t] = beta[0][t] + beta[1][t] * 2 + beta[2][t] * 2;
						
					}
				}
				
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}

void geomitry(IplImage*img,int i,int j) {
	int y[24] = { -2,-2,-2,-2,-2,-1,0,1,2,2,2,2,2,1,0,-1,-1,-1,-1,0,1,1,1,0 };
	int x[24] = { -2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2 ,-1,0,1,1,1,0,-1,-1 };
	//int y[8] = { -1,-1,-1,2,1,1,1,0 };
	//int x[8] = { -1,0,1,1,1,0,-1,-1 };

	double sum[3] = { 1,1,1 };
	for (int t = 0; t < 24; t++) {
		int iy = y[t] + i;
		int jx = x[t] + j;
		if (iy >= 0 && iy < img->height && jx >= 0 && jx < img->width) {
			CvScalar cv = cvGet2D(img, iy, jx);
			for (int k = 0; k < img->nChannels; k++) {
				//if(cv.val[k] != 0)
					sum[k] *= cv.val[k];
			}
		}
	}
	for (int k = 0; k < 3; k++) {
		sum[k] = pow(sum[k], 1 / 25);
	}
	CvScalar temp = cvGet2D(img, i, j);
	for (int k = 0; k < img->nChannels; k++) {
		temp.val[k] = sum[k];
	}
	cvSet2D(img, i, j, temp);
}

IplImage* GeometryMeanFilter(IplImage* src)

{

	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);

	int row, col;

	int h = src->height;

	int w = src->width;

	int pixels[3] = { 0,0,0 };
	int sum[3] = { 0,0,0 };

	int mn;

	//计算每个像素的去噪后color值
	int t;
	for (int i = 0; i<src->height; i++) {

		for (int j = 0; j<src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] != 0)
					break;
			}

			if (t >= src->nChannels) {
				sum[0] = sum[1] = sum[2] = 1;
				pixels[0] = pixels[1] = pixels[2] = 0;
				for (int m = -2; m <= 2; m++) {
					row = m + i;
					for (int n = -2; n <= 2; n++) {
						col = n + j;
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							CvScalar cs = cvGet2D(src, row, col);
							for (int k = 0; k < src->nChannels; k++) {
								if (cs.val[k] != 0) {
									pixels[k]++;
									sum[k] *= cs.val[k];
								}
							}
						
						}
					}
				}
				for (int k = 0; k < src->nChannels; k++) {
					if (pixels[k] > 0) {
						cs.val[k] = pow(sum[k], 1 / pixels[k]);
					}
				}
			}
			cvSet2D(dst, i, j, cs);

		}

	}

	return dst;

}




struct WINDOWS {
	CvScalar item;
	int i;
	int j;
};

int comp(const void*a, const void*b)
{
	return *(int*)a - *(int*)b;
}

int norm1(CvScalar a, CvScalar b,int channels) {
	int sum = 0;
	for (int i = 0; i < channels; i++) {
		sum += abs(a.val[i] - a.val[2]);
	}
	return sum;
}


IplImage* VectorMF(IplImage*src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	CvScalar *windows = new CvScalar(30);
	int t;
	int row, col;
	int rowrecord, colrecord;//记录是取哪个点的值
	int *rowwindow = new int(24);//记录每个窗口的坐标
	int	*colwindow = new int(24);
	int count = 0;
	CvScalar res;
	bool flag;
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++)
				if (cs.val[t] == 0)
					flag = true;
			if (flag) {
				count = 0;
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							rowwindow[count] = row;
							colwindow[count] = col;
							windows[count++] = cvGet2D(src, row, col);
						}
					}
				}
				int value = 999999;
				for (t = 0; t < count; t++) {
					for (int fuck = t + 1; fuck < count; fuck++) {
						int getValue = norm1(windows[t], windows[fuck], src->nChannels);
						if (getValue < value) {
							value = getValue;
							rowrecord = rowwindow[t];
							colrecord = colwindow[t];
							cs = windows[t];
						}
					}
				}
				
			}
			cvSet2D(dst, i, j, cs);
		}
	}

	return dst;
}

double calc(IplImage*src) {

	int noise = 0;
	int sum = 0;
	int t;
	for (int j = 0; j < src->width; j++) {
		CvScalar cs = cvGet2D(src, 0, j);
		if (cs.val[1] == 0 || cs.val[0] == 0 || cs.val[2] == 0)
			noise++;
		sum++;
	}
	return  1.0*noise / (1.0* sum);
}

IplImage* simple(IplImage*src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	int row, col;
	int t;
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] != 0)
					break;
			}
			if (cs.val[0] == 0) {
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							CvScalar cv = cvGet2D(src, row, col);
							for (int t = 0; t < src->nChannels; t++) {
								if (cv.val[t] != 0)
									cs.val[t] = 255;
							}
						}
					}
				}
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}

IplImage* MedianFilter(IplImage*src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	int t;
	int row, col;
	int sum[3] = { 0,0,0 };
	int pixels[3] = { 0,0,0 };
	bool flag = false;
	int last = -1;
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++)
				if (cs.val[t] == 0)
					flag = true;
			if (flag) {
				sum[0] = sum[1] = sum[2] = 0;
				pixels[0] = pixels[1] = pixels[2] = 0;
				int count = 0;
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;

						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							CvScalar cv = cvGet2D(src, row, col);
							for (int k = 0; k < src->nChannels; k++) {

								if (cv.val[k] != 0) {
									pixels[k] ++;
									sum[k] += cv.val[k];
								}
							}
						}
					}
				}
				for (int k = 0; k < src->nChannels; k++) {


					if (cs.val[k] == 0 && pixels[k] != 0) {
						cs.val[k] = sum[k] / pixels[k];
					}
					else {
						if (j + 1 < src->width) {
							CvScalar temp = cvGet2D(src, i, j + 1);
							cs.val[k] = temp.val[k];
						}
						else {
							CvScalar temp = cvGet2D(src, i, j - 1);
							cs.val[k] = temp.val[k];
						}
					}
				}
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}


IplImage* Fusion(IplImage*img1, IplImage*img2) {
	IplImage* dst = cvCreateImage(cvGetSize(img1), img1->depth, img1->nChannels);
	 
	for (int i = 0; i < img1->height; i++) {
		for (int j = 0; j < img2->width; j++) {
			CvScalar cs1 = cvGet2D(img1, i, j);
			CvScalar cs2 = cvGet2D(img2, i, j);
			CvScalar cs = cvGet2D(dst, i, j);
			for (int k = 0; k < img1->nChannels; k++) {
				cs.val[k] = (cs1.val[k] *1+ cs2.val[k]*0.0);
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}

void HSI(IplImage*src) {
	int step, step_hsi, channels, cd, cdhsi, b, g, r;

	uchar *data, *data_i, *data_s, *data_h;
	int i, j;
	double min_rgb, add_rgb, theta, den, num;

	IplImage* hsi_i = cvCreateImage(cvGetSize(src), src->depth, 1);    //创建亮度图象
	IplImage* hsi_s = cvCreateImage(cvGetSize(src), src->depth, 1);    //创建饱和度图象
	IplImage* hsi_h = cvCreateImage(cvGetSize(src), src->depth, 1);    //创建色彩图象

	step = src->widthStep;                                      //存储同列相邻行之间的比特数
	channels = src->nChannels;
	data = (uchar *)src->imageData;                                      //存储指向图象数据的指针

	step_hsi = hsi_i->widthStep;                                  //亮度图象的相邻行之间的比特


	data_i = (uchar *)hsi_i->imageData;                                    //存储指向子图象的数据指针

	data_s = (uchar *)hsi_s->imageData;

	data_h = (uchar *)hsi_h->imageData;




	for (i = 0; i < src->height; i++)
		for (j = 0; j < src->width; j++) {

			cd = i*step + j*channels;                                                //计算取元图象数据的位置
			cdhsi = i*step_hsi + j;                                                  //计算子图象数据存储的位置
			b = data[cd], g = data[cd + 1], r = data[cd + 2];
			data_i[cdhsi] = (int)((r + g + b) / 3);                                  //计算亮度子图象
			min_rgb = __min(__min(r, g), b);                                           //取最小值运算
			add_rgb = r + g + b;
			data_s[cdhsi] = (int)(255 - 765 * min_rgb / add_rgb);                       //饱和度S的范围显示为0～255，便于显示

			num = 0.5*((r - g) + (r - b));                                           //下面的式子计算图象的色彩H
			den = sqrt((double)((r - g)*(r - g) + (r - b)*(g - b)));


			if (0 == den)
				den = 0.01;
			theta = acos(num / den);

			if (b <= g)
				data_h[cdhsi] = (int)(theta * 255 / (2 * 3.14));
			else
				data_h[cdhsi] = (int)(255 - theta * 255 / (2 * 3.14));

			if (data_s[cdhsi] == 0)
				data_h[cdhsi] = 0;
		}
	int t = 0;
	int row;
	int col;
	int mixHsi = 0;
	for (int i = 0; i<src->height; i++) {

		for (int j = 0; j<src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] != 0)
					break;
			}
			if (t >= src->nChannels) {
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							
						}
					}
				}
			}
		}

	}
	

}

IplImage* MSMF(IplImage*src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	return dst;
}

IplImage* Demo(IplImage*src) {
	int row;
	int col;
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	double res = 0;
	int t;
	bool flag;
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] == 0)
					flag = true;
			}
			if (flag) {
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						CvScalar cs = cvGet2D(src, row, col);
						if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
							for (int k = 0; k < src->nChannels; k++) {
								if (cs.val[k] != 0) {

								}
							}
						}
					}
				}
				for (int k = 0; k < src->nChannels; k++) {
					if (cs.val[k] == 0) {

					}
				}
			}
			cvSet2D(dst, i, j, cs);
		}
	}
	return dst;
}


int main() {
	
	IplImage* img = cvLoadImage("data/B.png", -1);
	cvShowImage("origin", img);
	bool flag = true;
	int k;
	
	
	int t;
	int max = 0;
	CvScalar test;

	IplImage*res = GM(0.9,img);

	////IplImage* img2 = GeometryMeanFilter(img);
	////img = Fusion(img1, img2);
	cvShowImage("after", res);
	cvSaveImage("res_B.png", res);
	waitKey(0);
	//HSI(img);
}