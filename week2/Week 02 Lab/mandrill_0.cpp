/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: RGB explicit thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() { 

  // Read image from file
  Mat image = imread("mandrill1.jpg", 1);


  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     uchar pixelBlue = image.at<Vec3b>(y,x)[0];
     uchar pixelGreen = image.at<Vec3b>(y,x)[1];
     uchar pixelRed = image.at<Vec3b>(y,x)[2];

    image.at<Vec3b>(y,x)[0] = pixelRed;
    image.at<Vec3b>(y,x)[1] = pixelBlue;
    image.at<Vec3b>(y,x)[2] = pixelGreen;


 } }

  //Save thresholded image
  imwrite("mandrill1_re.jpg", image);

  //construct a window for image display
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);
   
  //visualise the loaded image in the window
  imshow("Display window", image);

  //wait for a key press until returning from the program
  waitKey(0);
  return 0;
}