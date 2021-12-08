/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame, string fileName);
float get_iou(Rect truth, Rect face);
int find_the_no_entry_num(vector<Rect> no_entry, Rect truths);
float caculate_f1_score(int TP, int TN, int FP, int FN);
void sobel(Mat &input, Mat &x_sobel, Mat &y_sobel, Mat &m_sobel, Mat &d_sobel);
vector<Vec3f> hough_circle_transform(Mat &input, int r_min, int r_max, Mat &direction, int min_distence, int hough_threshold);
// vector<Rect> no_entry_filter(vector<Rect> no_entry, vector<Vec3f> circles);
vector<Rect> no_entry_filter(Mat frame, vector<Rect> no_entry_VJ, vector<Vec3f> circles, vector<Rect> output);

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;

Rect GT_No_Entry0_1(155, 240, 50, 50);
Rect GT_No_Entry0_2(600, 240, 50, 50);

Rect GT_No_Entry1_1(170, 100, 75, 105);

Rect GT_No_Entry2_1(360, 140, 140, 120);

Rect GT_No_Entry3_1(20, 1, 90, 80);
Rect GT_No_Entry3_2(415, 10, 170, 180);

Rect GT_No_Entry4_1(195, 175, 40, 55);
Rect GT_No_Entry4_2(745, 205, 35, 50);

Rect GT_No_Entry5_1(125, 315, 35, 40);
Rect GT_No_Entry5_2(305, 330, 30, 35);
Rect GT_No_Entry5_3(430, 340, 30, 35);
Rect GT_No_Entry5_4(520, 350, 25, 35);
Rect GT_No_Entry5_5(593, 360, 20, 28);
Rect GT_No_Entry5_6(650, 370, 18, 22);
Rect GT_No_Entry5_7(692, 370, 20, 22);
Rect GT_No_Entry5_8(731, 380, 15, 15);
Rect GT_No_Entry5_9(845, 380, 10, 20);
Rect GT_No_Entry5_10(865, 385, 15, 15);

Rect GT_No_Entry6_1(180, 320, 140, 140);
Rect GT_No_Entry6_2(370, 210, 210, 210);
Rect GT_No_Entry6_3(590, 220, 180, 230);
Rect GT_No_Entry6_4(800, 410, 40, 130);

Rect GT_No_Entry7_1(555, 175, 35, 35);

Rect GT_No_Entry8_1(195, 80, 100, 90);
Rect GT_No_Entry8_2(145, 180, 60, 60);
Rect GT_No_Entry8_3(105, 240, 40, 40);
Rect GT_No_Entry8_4(85, 260, 30, 30);

Rect GT_No_Entry9_1(470, 575, 50, 55);

Rect GT_No_Entry10_1(230, 130, 80, 80);
Rect GT_No_Entry10_2(500, 155, 90, 90);
Rect GT_No_Entry10_3(580, 170, 50, 70);

Rect GT_No_Entry11_1(50, 215, 60, 60);
Rect GT_No_Entry11_2(440, 160, 90, 85);

Rect GT_No_Entry12_1(95, 240, 80, 80);
Rect GT_No_Entry12_2(245, 185, 45, 45);
Rect GT_No_Entry12_3(295, 120, 30, 40);
Rect GT_No_Entry12_4(435, 85, 25, 30);
Rect GT_No_Entry12_5(660, 120, 30, 40);
Rect GT_No_Entry12_6(815, 250, 80, 80);

Rect GT_No_Entry13_1(380, 575, 50, 50);

Rect GT_No_Entry14_1(430, 290, 60, 55);

Rect GT_No_Entry15_1(325, 125, 55, 55);
Rect GT_No_Entry15_2(385, 120, 60, 60);

/** @function main */
int main(int argc, const char **argv)
{
	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if (!cascade.load(cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	};

	string fileName = argv[1];

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame, fileName);

	// 4. Save Result Image
	imwrite("detected.jpg", frame);

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, string fileName)
{
	std::vector<Rect> no_entry_VJ;
	// std::vector<Rect> no_entry_finded;
	// std::vector<Rect> truth;
	Mat frame_gray;
	int TP;
	int TN;
	int FP;
	int FN;
	int num_GT;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale(frame_gray, no_entry_VJ, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300, 300));

	// 3. Print number of Faces found
	std::cout << no_entry_VJ.size() << std::endl;

	Mat dst;
	GaussianBlur(frame, dst, Size(3, 3), 1, 1, BORDER_DEFAULT);
	// medianBlur(frame, dst, 3);
	cvtColor(dst, dst, CV_BGR2GRAY);

	// imshow("GB", dst);

	Mat sobel_x(frame.size(), CV_32FC1);
	Mat sobel_y(frame.size(), CV_32FC1);
	Mat sobel_m(frame.size(), CV_32FC1);
	Mat sobel_d(frame.size(), CV_32FC1);

	sobel(dst, sobel_x, sobel_y, sobel_m, sobel_d);

	normalize(sobel_m, sobel_m, 0, 255, NORM_MINMAX, CV_8UC1);
	// normalize(sobel_d,sobel_d, 0, 255, NORM_MINMAX, CV_8UC1);

	cout << "check 1 " << endl;
	// threshold
	dst = sobel_m;
	for (int x = 0; x < dst.rows; x++)
	{
		for (int y = 0; y < dst.cols; y++)
		{
			if (dst.at<uchar>(x, y) > 70)
			{
				dst.at<uchar>(x, y) = 255;
			}
			else
			{
				dst.at<uchar>(x, y) = 0;
			}
		}
	};
	// imshow("img_m",dst);
	// imshow("img_d",sobel_d);
	// imshow("sobel_image_x&y", dst);

	cout << "check 2 " << endl;

	vector<Vec3f> pcircles;
	// HoughCircles(dst, pcircles, CV_HOUGH_GRADIENT,1, 80, 100, 50, 0, 150);
	pcircles = hough_circle_transform(dst, 0, 150, sobel_d, 1, 13);
	cout << "check 3 " << endl;

	// for (size_t i = 0; i < pcircles.size(); i++){
	//     Vec3f c = pcircles[i];
	//     circle(frame, Point(c[0], c[1]), c[2],Scalar(255, 0, 0),2 );
	// }
	for (int i = 0; i < no_entry_VJ.size(); i++)
	{
		rectangle(frame, Point(no_entry_VJ[i].x, no_entry_VJ[i].y), Point(no_entry_VJ[i].x + no_entry_VJ[i].width, no_entry_VJ[i].y + no_entry_VJ[i].height), Scalar(0, 255, 255), 2);
	};
	cout << "check 4 " << endl;
	vector<Rect> no_entry;
	no_entry = no_entry_filter(frame, no_entry_VJ, pcircles, no_entry);
	// no_entry = no_entry_VJ;
	cout << "check 5 " << endl;

	// 4. Draw box around faces found
	for (int i = 0; i < no_entry.size(); i++)
	{
		rectangle(frame, Point(no_entry[i].x, no_entry[i].y), Point(no_entry[i].x + no_entry[i].width, no_entry[i].y + no_entry[i].height), Scalar(0, 255, 0), 2);
	};

	cout << "check 6 " << endl;

	if (fileName == "No_entry/NoEntry0.bmp")
	{
		cv::rectangle(frame, GT_No_Entry0_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry0_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry0_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry0_2);
		TP = TP_1 + TP_2;
		num_GT = 2;
	};
	if (fileName == "No_entry/NoEntry1.bmp")
	{
		cv::rectangle(frame, GT_No_Entry1_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_no_entry_num(no_entry, GT_No_Entry1_1);
		num_GT = 1;
	};

	if (fileName == "No_entry/NoEntry2.bmp")
	{
		cv::rectangle(frame, GT_No_Entry2_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_no_entry_num(no_entry, GT_No_Entry2_1);
		num_GT = 1;
	};

	if (fileName == "No_entry/NoEntry3.bmp")
	{
		cv::rectangle(frame, GT_No_Entry3_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry3_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry3_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry3_2);
		TP = TP_1 + TP_2;
		num_GT = 2;
	};

	if (fileName == "No_entry/NoEntry4.bmp")
	{
		cv::rectangle(frame, GT_No_Entry4_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry4_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry4_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry4_2);
		TP = TP_1 + TP_2;
		num_GT = 2;
	};

	if (fileName == "No_entry/NoEntry5.bmp")
	{
		cv::rectangle(frame, GT_No_Entry5_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_2, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_3, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_4, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_5, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_6, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_7, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_8, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_9, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry5_10, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry5_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry5_2);
		int TP_3 = find_the_no_entry_num(no_entry, GT_No_Entry5_3);
		int TP_4 = find_the_no_entry_num(no_entry, GT_No_Entry5_4);
		int TP_5 = find_the_no_entry_num(no_entry, GT_No_Entry5_5);
		int TP_6 = find_the_no_entry_num(no_entry, GT_No_Entry5_6);
		int TP_7 = find_the_no_entry_num(no_entry, GT_No_Entry5_7);
		int TP_8 = find_the_no_entry_num(no_entry, GT_No_Entry5_8);
		int TP_9 = find_the_no_entry_num(no_entry, GT_No_Entry5_9);
		int TP_10 = find_the_no_entry_num(no_entry, GT_No_Entry5_10);
		TP = TP_1 + TP_2 + TP_3 + TP_4 + TP_5 + TP_6 + TP_7 + TP_8 + TP_9 + TP_10;
		num_GT = 10;
	};

	if (fileName == "No_entry/NoEntry6.bmp")
	{
		cv::rectangle(frame, GT_No_Entry6_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry6_2, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry6_3, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry6_4, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry6_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry6_2);
		int TP_3 = find_the_no_entry_num(no_entry, GT_No_Entry6_3);
		int TP_4 = find_the_no_entry_num(no_entry, GT_No_Entry6_4);
		TP = TP_1 + TP_2 + TP_3 + TP_4;
		num_GT = 4;
	};

	if (fileName == "No_entry/NoEntry7.bmp")
	{
		cv::rectangle(frame, GT_No_Entry7_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_no_entry_num(no_entry, GT_No_Entry7_1);
		num_GT = 1;
	};

	if (fileName == "No_entry/NoEntry8.bmp")
	{
		cv::rectangle(frame, GT_No_Entry8_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry8_2, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry8_3, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry8_4, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry8_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry8_2);
		int TP_3 = find_the_no_entry_num(no_entry, GT_No_Entry8_3);
		int TP_4 = find_the_no_entry_num(no_entry, GT_No_Entry8_4);
		TP = TP_1 + TP_2 + TP_3 + TP_4;
		num_GT = 4;
	};

	if (fileName == "No_entry/NoEntry9.bmp")
	{
		cv::rectangle(frame, GT_No_Entry9_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_no_entry_num(no_entry, GT_No_Entry9_1);
		num_GT = 1;
	};

	if (fileName == "No_entry/NoEntry10.bmp")
	{
		cv::rectangle(frame, GT_No_Entry10_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry10_2, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry10_3, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry10_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry10_2);
		int TP_3 = find_the_no_entry_num(no_entry, GT_No_Entry10_3);
		TP = TP_1 + TP_2 + TP_3;
		num_GT = 3;
	};

	if (fileName == "No_entry/NoEntry11.bmp")
	{
		cv::rectangle(frame, GT_No_Entry11_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry11_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry11_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry11_2);
		TP = TP_1 + TP_2;
		num_GT = 2;
	};

	if (fileName == "No_entry/NoEntry12.bmp")
	{
		cv::rectangle(frame, GT_No_Entry12_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry12_2, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry12_3, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry12_4, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry12_5, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry12_6, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry12_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry12_2);
		int TP_3 = find_the_no_entry_num(no_entry, GT_No_Entry12_3);
		int TP_4 = find_the_no_entry_num(no_entry, GT_No_Entry12_4);
		int TP_5 = find_the_no_entry_num(no_entry, GT_No_Entry12_5);
		int TP_6 = find_the_no_entry_num(no_entry, GT_No_Entry12_6);
		TP = TP_1 + TP_2 + TP_3 + TP_4 + TP_5 + TP_6;
		num_GT = 6;
	};

	if (fileName == "No_entry/NoEntry13.bmp")
	{
		cv::rectangle(frame, GT_No_Entry13_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_no_entry_num(no_entry, GT_No_Entry13_1);
		num_GT = 1;
	};

	if (fileName == "No_entry/NoEntry14.bmp")
	{
		cv::rectangle(frame, GT_No_Entry14_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_no_entry_num(no_entry, GT_No_Entry14_1);
		num_GT = 1;
	};

	if (fileName == "No_entry/NoEntry15.bmp")
	{
		cv::rectangle(frame, GT_No_Entry15_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry15_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_no_entry_num(no_entry, GT_No_Entry15_1);
		int TP_2 = find_the_no_entry_num(no_entry, GT_No_Entry15_2);
		TP = TP_1 + TP_2;
		num_GT = 2;
	};

	FP = no_entry.size() - TP;
	TN = 0;
	FN = num_GT - TP;
	float f1_score = caculate_f1_score(TP, TN, FP, FN);
	float TPR = (float)TP / ((float)TP + (float)FN);

	std::cout << "TP = " << TP << std::endl;
	std::cout << "FP = " << FP << std::endl;
	std::cout << "TN = " << TN << std::endl;
	std::cout << "FN = " << FN << std::endl;
	std::cout << "num_GT =" << num_GT << std::endl;
	std::cout << "f1 score = " << f1_score << std::endl;
	std::cout << "TPR =" << TPR << std::endl;

	imshow("frame", frame);
	waitKey(0);
}

float get_iou(Rect truth, Rect face)
{
	float width = min(face.x + face.width, truth.x + truth.width) - max(face.x, truth.x);
	float height = min(face.y + face.height, truth.y + truth.height) - max(face.y, truth.y);

	if (width <= 0 or height <= 0)
		return 0;

	float int_area = width * height;
	float uni_area = (face.width * face.height) + (truth.width * truth.height) - int_area;

	// std::cout << int_area/uni_area;
	return int_area / uni_area;
}

int find_the_no_entry_num(vector<Rect> no_entry, Rect truths)
{
	float iou_thredhold = 0.6;
	int TP = 0;
	for (size_t i = 0; i < no_entry.size(); i++)
	{
		float IoU = get_iou(truths, no_entry[i]);
		if (IoU > iou_thredhold)
		{
			// std::cout << IoU << std::endl;
			TP++;
		}
	}
	// std::cout << TP << std::endl;
	return TP;
}

float caculate_f1_score(int TP, int TN, int FP, int FN)
{

	float tp = static_cast<float>(TP);
	float tn = static_cast<float>(TN);
	float fp = static_cast<float>(FP);
	float fn = static_cast<float>(FN);

	float f1_score = 2 * tp / (2 * tp + fp + fn);
	// float f1_score = 4.0f/11.0f
	return f1_score;
}
void sobel(Mat &input, Mat &x_sobel, Mat &y_sobel, Mat &m_sobel, Mat &d_sobel)
{

	Mat X_kernel = Mat::ones(3, 3, CV_32F);
	Mat Y_kernel = Mat::ones(3, 3, CV_32F);

	X_kernel.at<float>(0, 0) = -1;
	X_kernel.at<float>(1, 0) = -2;
	X_kernel.at<float>(2, 0) = -1;
	X_kernel.at<float>(0, 1) = 0;
	X_kernel.at<float>(1, 1) = 0;
	X_kernel.at<float>(2, 1) = 0;
	X_kernel.at<float>(0, 2) = 1;
	X_kernel.at<float>(1, 2) = 2;
	X_kernel.at<float>(2, 2) = 1;

	Y_kernel.at<float>(0, 0) = -1;
	Y_kernel.at<float>(1, 0) = 0;
	Y_kernel.at<float>(2, 0) = 1;
	Y_kernel.at<float>(0, 1) = -2;
	Y_kernel.at<float>(1, 1) = 0;
	Y_kernel.at<float>(2, 1) = 2;
	Y_kernel.at<float>(0, 2) = -1;
	Y_kernel.at<float>(1, 2) = 0;
	Y_kernel.at<float>(2, 2) = 1;

	Mat input_padded;
	copyMakeBorder(input, input_padded, 1, 1, 1, 1, BORDER_REPLICATE);

	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			float x_sum = 0.0;
			float y_sum = 0.0;
			for (int a = -1; a <= 1; a++)
			{
				for (int b = -1; b <= 1; b++)
				{
					float val_img = (int)input_padded.at<uchar>(i + a + 1, j + b + 1);
					float x_kernel = X_kernel.at<float>(a + 1, b + 1);
					float y_kernel = Y_kernel.at<float>(a + 1, b + 1);

					x_sum = x_sum + val_img * x_kernel;
					y_sum = y_sum + val_img * y_kernel;
				}
			}
			x_sobel.at<float>(i, j) = (float)x_sum;
			y_sobel.at<float>(i, j) = (float)y_sum;
			m_sobel.at<float>(i, j) = (float)sqrt((y_sum * y_sum) + (x_sum * x_sum));
			d_sobel.at<float>(i, j) = (float)atan2(y_sum, x_sum);
		}
	}
	cout << "sobel edge detection finished" << endl;
}

vector<Vec3f> hough_circle_transform(Mat &input, int r_min, int r_max, Mat &direction, int min_distence, int hough_threshold)
{

	int ***hough_space = (int ***)malloc(input.rows * sizeof(int **));
	for (int i = 0; i < input.rows; i++)
	{
		hough_space[i] = (int **)malloc(input.cols * sizeof(int *));
		for (int j = 0; j < input.cols; j++)
		{
			hough_space[i][j] = (int *)malloc(r_max * sizeof(int));
		}
	}

	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			for (int r = 0; r < r_max; r++)
			{
				hough_space[i][j][r] = 0;
			}
		}
	}

	for (int x = 0; x < input.rows; x++)
	{
		for (int y = 0; y < input.cols; y++)
		{
			if (input.at<uchar>(x, y) == 255)
			{
				for (int r = 0; r < r_max; r++)
				{
					int x_circle = int(r * sin(direction.at<float>(x, y)));
					int y_circle = int(r * cos(direction.at<float>(x, y)));

					if ((x - x_circle) >= 0 && (x - x_circle) < input.rows && (y - y_circle) >= 0 && (y - y_circle) < input.cols)
					{
						hough_space[(x - x_circle)][(y - y_circle)][r] = hough_space[(x - x_circle)][(y - y_circle)][r] + 1;
					}
					if ((x + x_circle) >= 0 && (x + x_circle) < input.rows && (y + y_circle) >= 0 && (y + y_circle) < input.cols)
					{
						hough_space[(x + x_circle)][(y + y_circle)][r] = hough_space[(x + x_circle)][(y + y_circle)][r] + 1;
					}
				}
			}
		}
	}

	Mat hough_output(input.rows, input.cols, CV_32FC1);

	for (int x = 0; x < input.rows; x++)
	{
		for (int y = 0; y < input.cols; y++)
		{
			for (int r = r_min; r < r_max; r++)
			{
				hough_output.at<float>(x, y) += hough_space[x][y][r];
			}
		}
	}
	imwrite("hough.jpg", hough_output);

	vector<Vec3f> circles;
	for (int x = 0; x < input.rows; x++)
	{
		for (int y = 0; y < input.cols; y++)
		{
			bool test_pass = true;
			map<int, int> t_circles;
			for (int r = r_min; r < r_max; r++)
			{
				if (hough_space[x][y][r] > hough_threshold)
				{
					t_circles[r] = hough_space[x][y][r];
				}
			}
			int max_c = 0;
			int max_r = 0;

			for (map<int, int>::const_iterator it = t_circles.begin(); it != t_circles.end(); ++it)
			{

				for (int i = 0; i < circles.size(); i++)
				{
					Vec3f circle = circles[i];
					int r = circle[2];
					if (r - min_distence < it->first && r + min_distence > it->first)
					{
						test_pass = false;
					}
				}
				if (test_pass)
				{
					Vec3f circle(y, x, it->first);
					circles.push_back(circle);
				}
			}
		}
	}

	return circles;
}

vector<Rect> no_entry_filter(Mat frame, vector<Rect> no_entry_VJ, vector<Vec3f> circles, vector<Rect> output)
{
	float iou_thredhold = 0.4;
	vector<Rect> circles_rects;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3f c = circles[i];
		Rect circles_rect(c[0] - c[2], c[1] - c[2], 2 * c[2], 2 * c[2]);
		circles_rects.push_back(circles_rect);
	}
	cout << "filter_check 1 " << endl;

	for (size_t i = 0; i < circles_rects.size(); i++)
	{
		rectangle(frame, Point(circles_rects[i].x, circles_rects[i].y), Point(circles_rects[i].x + circles_rects[i].width, circles_rects[i].y + circles_rects[i].height), Scalar(255, 0, 0), 2);
	}
	cout << "filter_check 2 " << endl;

	for (size_t i = 0; i < no_entry_VJ.size(); i++)
	{
		for (size_t j = 0; j < circles_rects.size(); j++)
		{
			float IoU = get_iou(circles_rects[j], no_entry_VJ[i]);
			if (IoU > iou_thredhold)
			{
				output.push_back(no_entry_VJ[i]);
			}
		}
	}
	return output;
	cout << "filter_check 3 " << endl;
}

// ./a.out No_entry/NoEntry1.bmp
// g++ subtask_3.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4

