#include <opencv.hpp>
#include <cv.h>

using namespace cv;
#define N 48
bool GetMatrixInverse(double src[N][N], int n, double des[N][N]);
double getA(double arcs[N][N], int n);
void  getAStart(double arcs[N][N], int n, double ans[N][N]);


void InverseMatrix(double A[3][3], int n, double B[3][3])
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



void matrixmultiply(double a[][N], double b[][N], int row, int col, double dst[][N]) {

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
	double a[N][N];
	double b[N][N];
	double res[N][N];
	double beta[N][N];
	double inverse[N][N];

	double inversey[N][N];
	double y[N][N];
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] == 0)
					flag = true;
			}
			if (flag) {

				for (t = 0; t < src->nChannels; t++) {
					count = 0;
					if (cs.val[t] == 0) {
						for (int m = -2; m <= 2; m++) {

							for (int n = -2; n <= 2; n++) {
								row = i + m;
								col = j + n;

								if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
									CvScalar cv = cvGet2D(src, row, col);
									if (cv.val[t] != 0) {
										a[count][0] = 1;
										a[count][1] = row;
										a[count][2] = col;
										
										y[count][t] = cv.val[t];
										count++;
									}

								}
							}
						}
						for (int k = 0; k < count; k++) {
							b[0][k] = 1;
							b[1][k] = a[k][1];
							b[2][k] = a[k][2];
						}
						//x_ 乘以x得到res矩阵
						matrixmultiply(b, a, 3, count, res);
						//res矩阵求逆
						bool isA = GetMatrixInverse(res, 3, inverse);
						if (isA) {
							//x_和像素值矩阵相乘得到inversey
							matrixmultiply(b, y, 3, count, inversey);
							//inverse和inversey相乘得到beta
							matrixmultiply(inverse, inversey, 3, src->nChannels, beta);
							cs.val[t] = beta[0][t] + beta[1][t]*i+ beta[2][t] * j;
						}
						else {
							CvScalar cleft;
							for (int left = 0; i - left >= 0; left++) {
								cleft = cvGet2D(src, i - left, j);
								if (cleft.val[t] != 0)
									break;
							}
							CvScalar cright;
							for (int right = 0; right + i < src->width; right++) {
								cright = cvGet2D(src, right + i, j);
								if (cright.val[t] != 0)
									break;
							}
							cs.val[t] = (cleft.val[t] + cright.val[t]) / 2;
						}
							

					
							

					}
				}

			}
			cvSet2D(dst, i, j, cs);
		}
		
	}
	return dst;
}

double norm(double a[2], double b[2]) {
	return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2));
}

double RBFKernel(double a[2], double b[2]) {
	double gama = 10;

	return exp(pow(norm(a, b),2)/(pow(2*gama,2)));
}


IplImage *GaussianRegression(IplImage*src) {
	int row;
	int col;
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	int t;
	bool flag;
	double points[N][2];
	double pixels[N][N];
	double K[N][N];
	double res[N][N];
	double Kinverse[N][N];
	double avg[N][N];
	int count = 0;
	bool out;
	int max = 0;
	int min;
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			CvScalar cs = cvGet2D(src, i, j);
			flag = false;
			for (t = 0; t < src->nChannels; t++) {
				if (cs.val[t] == 0) {
					flag = true;
					break;
				}
					
			}
			if (flag) {
				out = false;
				for (int t = 0; t < src->nChannels; t++) {
					max = 0;
					count = 0;
					min = 999999;
					for (int m = -2; m <= 2; m++) {
						if (out)
							break;
						row = i + m;
						for (int n = -2; n <= 2; n++) {
							if (out)
								break;
							col = j + n;
							
							if (row >= 0 && row < src->height&&col >= 0 && col < src->width) {
								CvScalar cv = cvGet2D(src, row, col);
								if (cv.val[t] != 0) {
									points[count][0] = row;
									points[count][1] = col;
									pixels[count][0] = cv.val[t];
									if (max < pixels[count][0])
										max = pixels[count][0];
									if (min > pixels[count][0])
										min = pixels[count][0];
									count++;
									if (count >= 4)
									{
										out = true;
									}
								}
							}
						}
					}
					//高斯回归，协方差矩阵构建
					for (int num = 0; num < count; num++) {
						for (int innum = num; innum < count; innum++) {
							if (num == innum)
								K[num][num] = 1.7;
							else {
								//RBF核函数
								K[num][innum] = K[innum][num] = RBFKernel(points[num], points[innum]);
							}
						}
					}
					//计算预测点和其它点的核值
					double Kstar[N][N];
					double temp[2] = { i,j };
					for (int cur = 0; cur < count; cur++) {
						
							
							Kstar[0][cur] = RBFKernel(temp, points[cur]);
						
					}
					
					bool isA = GetMatrixInverse(K, count, Kinverse);
					if (isA) {
						matrixmultiply(Kstar, Kinverse, 1, count, res);
						matrixmultiply(res, pixels, 1, 1, avg);
						int temp = (int)avg[0][0];
						if (max < temp)
							cs.val[t] = max;
						else if (min > temp) {
							cs.val[t] = min;
						}
						else
							cs.val[t] = temp;
					}
					else {
						CvScalar cleft;
						int left;
						bool leftend = true;
						for (left = 0; i - left >= 0; left++) {
							cleft = cvGet2D(src, i - left, j);
							if (cleft.val[t] != 0) {
								leftend = false;
								break;
							}
								
						}
						CvScalar cright;
						int right;
						bool rightend = true;
						for (right = 0; right + i < src->width; right++) {
							cright = cvGet2D(src, right + i, j);
							if (cright.val[t] != 0) {
								rightend = false;
								break;
							}
						}
						if(!leftend && !rightend)
							cs.val[t] = (cleft.val[t] + cright.val[t]) / 2;
						else if (leftend) {
							cs.val[t] = cright.val[t];
						}
						else {
							cs.val[t] = cleft.val[t];
						}
					}
					
					cvSet2D(dst, i, j, cs);
				}
				
			}
			
		}
	}
	return dst;
}




int main() {

	IplImage* img = cvLoadImage("data/A.png", -1);
	cvShowImage("origin", img);
	bool flag = true;
	int k;


	int t;
	int max = 0;
	CvScalar test;

	IplImage*res = LinearRegression(img);

	cvShowImage("after", res);
	cvSaveImage("LinearB.png", res);
	waitKey(0);
}