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
void detectAndDisplay( Mat frame);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;
int GT_No_Entry1  [1][4] = {(230, 480, 75, 75)};
int GT_No_Entry2  [2][4] = {(30 , 410, 40, 40), 
	                        (300, 410, 30, 30)};
int GT_No_Entry4  [5][4] = {(170, 270, 50, 60),
	                        (480, 280, 50, 70),
						    (570, 270, 30, 30),
						    (670, 280, 60, 60),
						    (870, 240, 60, 60)};
int GT_No_Entry5  [1][4] = {(790, 290, 60, 60)};
int GT_No_Entry7  [2][4] = {(880, 220, 70, 70),
	                        (380, 220, 20, 20)};
int GT_No_Entry11 [2][4] = {(670, 400, 30, 30),
	                        (750, 400, 30, 30)};


/** @function main */
int main( int argc, const char** argv )
{
    
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    String fileName = argv[1];
	printf (fileName);
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );




	if (fileName == "NoEntry1.bmp")
	{
		cv::rectangle(frame, Rect(230,480,75,75), cv::Scalar(0, 0, 255), 2);
	};
	if (fileName == "NoEnry2.bmp")
	{
		for(int i = 0; i <= 1; i ++)
		{
		    cv::rectangle(frame, Rect(GT_No_Entry1[i][0],GT_No_Entry1[i][1],GT_No_Entry1[i][2],GT_No_Entry1[i][3]), cv::Scalar(0, 0, 255), 2);
		} 
	}
	if (fileName == "NoEnry4.bmp")
	{
		for(int i = 0; i <= 3; i ++)
		{
		    cv::rectangle(frame, Rect(GT_No_Entry1[i][0],GT_No_Entry1[i][1],GT_No_Entry1[i][2],GT_No_Entry1[i][3]), cv::Scalar(0, 0, 255), 2);
		} 
	}
	if (fileName == "NoEnry5.bmp")
	{
		cv::rectangle(frame, Rect(GT_No_Entry1[0][0],GT_No_Entry1[0][1],GT_No_Entry1[0][2],GT_No_Entry1[0][3]), cv::Scalar(0, 0, 255), 2);
	}
	if (fileName == "NoEnry7.bmp")
	{
		for(int i = 0; i <= 1; i ++)
		{
		    cv::rectangle(frame, Rect(GT_No_Entry1[i][0],GT_No_Entry1[i][1],GT_No_Entry1[i][2],GT_No_Entry1[i][3]), cv::Scalar(0, 0, 255), 2);
		} 
	}
	if (fileName == "NoEnry11.bmp")
	{
		for(int i = 0; i <= 1; i ++)
		{
		    cv::rectangle(frame, Rect(GT_No_Entry1[i][0],GT_No_Entry1[i][1],GT_No_Entry1[i][2],GT_No_Entry1[i][3]), cv::Scalar(0, 0, 255), 2);
		} 
	}





	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
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
}
