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
	}        //��ʼ��B=E

	for (i = 0; i<n; i++)
		for (j = 0; j<n; j++)
			a[i][j] = A[i][j];  //����A��a������ı�A��ֵ
	for (i = 0; i<n; i++)
		for (j = n; j<m; j++)
			a[i][j] = B[i][j - n];  //����B��a���������

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
	}        //˳���˹��ȥ�������½�Ϊ��

	for (i = 1; i <= n; i++)
	{
		temp = a[i - 1][i - 1];
		for (j = 1; j <= m; j++)
		{
			a[i - 1][j - 1] /= temp;
		}
	}        //��һ��

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
	}        //�����˹��ȥ��������������Ϊ��λ����

	for (i = 0; i<n; i++)
		for (j = 0; j<n; j++)
			B[i][j] = a[i][j + n];  //ȡ��������

	for (i = 0; i<n; i++)
		for (j = 0; j<n; j++)
			if (fabs(B[i][j])<0.0001)
				B[i][j] = 0.0;

	delete[]a;
}
//�õ���������src������󱣴浽des�С�
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

//����һ��չ������|A|
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

//����ÿһ��ÿһ�е�ÿ��Ԫ������Ӧ������ʽ�����A*
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
					out = false;
					count = 0;
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
						//x_ ����x�õ�res����
						matrixmultiply(b, a, 3, count, res);
						//res��������
						bool isA = GetMatrixInverse(res, 3, inverse);
						if (isA) {
							ok = true;
							//x_������ֵ������˵õ�inversey
							matrixmultiply(b, y, 3, count, inversey);
							//inverse��inversey��˵õ�beta
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






int main() {

	IplImage* img = cvLoadImage("data/B.png", -1);
	cvShowImage("origin", img);
	bool flag = true;
	int k;


	int t;
	int max = 0;
	CvScalar test;

	IplImage*res = LinearRegression(img);

	cvShowImage("after", res);
	cvSaveImage("tesB.png", res);
	waitKey(0);
}