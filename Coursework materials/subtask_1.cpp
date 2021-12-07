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
void detectAndDisplay( Mat frame, string fileName);
float get_iou(Rect truth, Rect face);
int find_the_face_num(vector<Rect> faces, Rect truths);
float caculate_f1_score (int TP, int TN, int FP, int FN);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

Rect GT_No_Entry1_1(230, 480, 75, 75);
Rect GT_No_Entry2_1(30 , 410, 40, 40);
Rect GT_No_Entry2_2(300, 410, 30, 30);
Rect GT_No_Entry4_1(170, 270, 50, 60);
Rect GT_No_Entry4_2(480, 280, 50, 70);
Rect GT_No_Entry4_3(572, 270, 35, 32);
Rect GT_No_Entry4_4(670, 280, 60, 60);
Rect GT_No_Entry4_5(870, 240, 65, 65);
Rect GT_No_Entry5_1(790, 290, 50, 50);
Rect GT_No_Entry7_1(880, 220, 70, 70);
Rect GT_No_Entry7_2(385, 220, 25, 25);
Rect GT_No_Entry11_1(670, 400, 30, 30);
Rect GT_No_Entry11_2(750, 400, 30, 30);


/** @function main */
int main( int argc, const char** argv )
{
    
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    string fileName = argv[1];
	// printf (fileName);
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, fileName );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, string fileName)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	int TP;
	int TN;
	int FP;
	int FN;
	int num_GT;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	} 

		if (fileName == "No_entry/NoEntry1.bmp")
	{
		cv::rectangle(frame, GT_No_Entry1_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_face_num(faces, GT_No_Entry1_1);
		num_GT = 1;
		// cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0),2);
	};
	if (fileName == "No_entry/NoEntry2.bmp")
	{
		cv::rectangle(frame, GT_No_Entry2_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry2_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_face_num(faces, GT_No_Entry2_1);
		int TP_2 = find_the_face_num(faces, GT_No_Entry2_1);
		TP = TP_1 + TP_2;
		num_GT = 2;
		// std::cout << TP;
	}
	if (fileName == "No_entry/NoEntry4.bmp")
	{
		cv::rectangle(frame, GT_No_Entry4_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry4_2, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry4_3, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry4_4, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry4_5, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_face_num(faces, GT_No_Entry4_1);
		int TP_2 = find_the_face_num(faces, GT_No_Entry4_2);
		int TP_3 = find_the_face_num(faces, GT_No_Entry4_3);
		int TP_4 = find_the_face_num(faces, GT_No_Entry4_4);
		int TP_5 = find_the_face_num(faces, GT_No_Entry4_5);
		TP = TP_1 + TP_2 + TP_3 + TP_4 + TP_5;
		num_GT  = 5;
	}
	if (fileName == "No_entry/NoEntry5.bmp")
	{
		cv::rectangle(frame, GT_No_Entry5_1, cv::Scalar(0, 0, 255), 2);
		TP = find_the_face_num(faces, GT_No_Entry5_1);
		num_GT = 1;
	}
	if (fileName == "No_entry/NoEntry7.bmp")
	{
		cv::rectangle(frame, GT_No_Entry7_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry7_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_face_num(faces, GT_No_Entry7_1);
		int TP_2 = find_the_face_num(faces, GT_No_Entry7_2);
		TP = TP_1 + TP_2;
		num_GT = 2;

	}
	if (fileName == "No_entry/NoEntry11.bmp")
	{
		cv::rectangle(frame, GT_No_Entry11_1, cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, GT_No_Entry11_2, cv::Scalar(0, 0, 255), 2);
		int TP_1 = find_the_face_num(faces, GT_No_Entry11_1);
		int TP_2 = find_the_face_num(faces, GT_No_Entry11_2);
		TP = TP_1 + TP_2;
		num_GT = 2;

	}
	FP = faces.size() - TP;
	TN = 0;
	// FN = max(0, num_GT - TP);
	FN = num_GT - TP;

	float f1_score = caculate_f1_score(TP, TN, FP, FN);
	float TPR = (float)TP/((float)TP + (float)FN);
 

 
	std::cout << "TP = " << TP << std::endl;
	std::cout << "FP = " << FP << std::endl;
	std::cout << "TN = " << TN << std::endl;
	std::cout << "FN = " << FN << std::endl;
	std::cout << "num_GT =" << num_GT << std::endl;
    std::cout << "f1 score = " << f1_score << std::endl;
	std::cout << "TPR =" << TPR << std::endl;

	// printf("TP = %f \n",TP);
	// printf("FP = %f \n",FP);
	// printf("TN = %d \n",TN);
	// printf("FN = %d \n",FN);
	// printf("f1 score = %f \n", f1_score);


}

float get_iou(Rect truth, Rect face) {
	float width = min(face.x + face.width, truth.x + truth.width) - max(face.x, truth.x);
	float height = min(face.y + face.height, truth.y + truth.height) - max(face.y, truth.y);

	if(width <= 0 or height <= 0) return 0;

	float int_area = width * height;
	float uni_area = (face.width * face.height) + (truth.width * truth.height) - int_area;

    // std::cout << int_area/uni_area;
	return int_area/uni_area;
}

int find_the_face_num(vector<Rect> faces, Rect truths){
	float iou_thredhold = 0.60;
	int TP = 0;
	for ( size_t i = 0; i < faces.size(); i++){
		float IoU = get_iou(truths, faces[i]);
		if( IoU > iou_thredhold){
			// std::cout << IoU << std::endl;
			TP ++;
		}
	}
	// std::cout << TP << std::endl;
	return TP;
}


float caculate_f1_score (int TP, int TN, int FP, int FN){

	float tp = static_cast<float>(TP);
	float tn = static_cast<float>(TN);
	float fp = static_cast<float>(FP);
	float fn = static_cast<float>(FN);

	// float tp = (float)TP;
	// float tn = (float)TN;
	// float fp = (float)FP;
	// float fn = (float)FN;

	// float P = TP/(TP + FP); // precision //
	// float R = TP/(TP + FN); // recall //

	// float f1_score = 2*R*P/(R+P);
	float f1_score = 2*tp/(2*tp + fp + fn);
	// float f1_score = 4.0f/11.0f
	return f1_score;
}
