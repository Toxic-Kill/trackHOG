#include <iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	//函数声明
	void calHOG(cv::Mat Mat1, float *hist, int dim, int nx, int ny);
	float calDis(float *hist1, float *hist2, int size, int count);

	//读取图像
	cv::Mat srcMat = cv::imread("D:\\Files\\img.png", 0);
	cv::Mat tagMat = cv::imread("D:\\Files\\template.png", 0);

	//判断读取成功性
	if (srcMat.empty() || tagMat.empty())
	{
		std::cout << "Can't open the image" << endl;
		return -1;
	}

	//设置参数
	int nx = tagMat.cols;
	int ny = tagMat.rows;
	int aglDim = 8;
	int x = srcMat.cols - nx + 1;
	int y = srcMat.rows - ny + 1;

	int bins = x * y * aglDim;

	float * src_hist = new float[bins*256];
	memset(src_hist, 0, sizeof(float)*bins*256);
	float * tag_hist = new float[bins*256];
	memset(tag_hist, 0, sizeof(float)*bins*256);
	float *dis = new float[x*y*8];
	memset(dis, 0, sizeof(float)*x*y*8);

	//计算图像HOG
	calHOG(srcMat, src_hist, aglDim, srcMat.cols, srcMat.rows);
	calHOG(tagMat, tag_hist, aglDim, nx, ny);

	
	for (int count = 0; count < x*y; count++)
	{
		dis[count] = calDis(src_hist, tag_hist, bins, count);
	}

	int min = dis[0];
	int n = 0;
	for (int m = 1; m < x*y; m++)
	{
		if (dis[m] < min)
		{
			min = dis[m];
			n = m;
		}
	}

	cv::Rect rect;
	rect.x = (nx + n) % nx;
	rect.y = (nx + n) / nx + 1;
	rect.width = nx;
	rect.height = ny;
	rectangle(srcMat, rect, CV_RGB(255, 0, 0), 1, 8, 0);

	cv::imshow("dst", srcMat);
	waitKey(0);

	//输出比较结果

	delete[] src_hist;
	delete[] tag_hist;
	delete[] dis;

	return 0;
}

//定义计算图像HOG的函数
void calHOG(cv::Mat Mat1, float *hist, int dim, int nx, int ny)
{
	//参数设计
	int nX = Mat1.cols - 108+1;
	int nY = Mat1.rows - 48+1;
	int binAngle = 360 / dim;

	//计算梯度与角度
	cv::Mat gx, gy;
	cv::Mat mag, angle;
	cv::Sobel(Mat1, gx, CV_32F, 1, 0, 1);
	cv::Sobel(Mat1, gy, CV_32F, 0, 1, 1);
	cv::cartToPolar(gx, gy, mag, angle, true);

	//遍历赋值
	int cellNum = 0;
	cv::Rect roi;
	roi.x = 0;
	roi.y = 0;
	roi.width = 108;
	roi.height = 48;

	for (int i = 0; i < nY; i++) {
		for (int j = 0; j < nX; j++) {
			cv::Mat roiMat;
			cv::Mat roiMag;
			cv::Mat roiAgl;

			roi.x = j;
			roi.y = i;

			//赋值图像
			roiMat = Mat1(roi);
			roiMag = mag(roi);
			roiAgl = angle(roi);

			//当前cell第一个元素在数组中的位置
			int head = (i*nx + j)*dim;

			for (int n = 0; n < roiMat.rows; n++) {
				for (int m = 0; m < roiMat.cols; m++) {
					//计算角度在哪个bin，通过int自动取整实现
					int pos = (int)(roiAgl.at<float>(n, m) / binAngle);
					hist[head + pos] += roiMag.at<float>(n, m);
				}
			}
		}
	}
}

//定义比较图像的函数
float calDis(float *hist1, float *hist2, int size, int count)
{
	float sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += (hist1[8 * count + i] - hist2[i])*(hist1[8 * count + i] - hist2[i]);
	}
	sum = sqrt(sum);
	return sum;
}