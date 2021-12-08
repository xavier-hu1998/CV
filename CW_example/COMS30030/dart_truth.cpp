// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>

#define pi 3.14159265358979323846

#define threshold_magnitude 75
#define threshold_circles 15
#define threshold_rho_space 5
#define threshold_lines 160
#define threshold_pixels_in_lines 20

using namespace std;
using namespace cv;

/** Function Headers */
Mat normalize_and_save(Mat input, string num, string name);

vector<Rect> read_csv(string num);
vector<Rect> filter_darts(vector<Rect> darts, vector<vector<int>> circles, string num, int pixels);
vector<Rect> filter_by_radius(vector<Rect> darts, float avg);
vector<Rect> group_darts(vector<Rect> darts);
vector<string> split( const string &line, char delimiter );
vector<vector<int>> ch_transform(Mat &input, int r_min, int r_max, Mat &direction, string num);

int pixel_count(string num);

float get_iou(Rect t, Rect d);
float get_f1_score(float t_p, float f_p, float f_n);
float average_radius(vector<vector<int>> circles);

void detectAndDisplay( Mat frame, vector<Rect> truths, string num, vector<vector<int>> circles );
void sobel(Mat &input, Mat &output_x, Mat &output_y, Mat &output_mag, Mat &output_dir);
void write_sobel(Mat x, Mat y, string num);
void threshold(Mat &input, int t, string num, string ver);
void gaussian(Mat &input, int size);
void filter_non_max(Mat &input_mag, Mat &input_dir);
void lh_transform(Mat &input, Mat &direction, string num);
void draw_circles(Mat &input, vector<vector<int> > circles, string num);
void draw(Mat frame, vector<Rect> truths, vector<Rect> darts, vector<Rect> darts_f, vector<vector<int>> circles, string num);
void debug_out(vector<Rect> darts_filtered, vector<Rect> truths, string num);

String cascade_name = "dart_cascade/cascade.xml";
CascadeClassifier cascade;

int **malloc2dArray(int dim1, int dim2) {
    int i, j;
    int **array = (int **) malloc(dim1 * sizeof(int *));
 
    for (i = 0; i < dim1; i++) {
        array[i] = (int *) malloc(dim2 * sizeof(int));
    }
    return array;
}
int ***malloc3dArray(int dim1, int dim2, int dim3) {
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));
 
    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	    for (j = 0; j < dim2; j++) {
  	        array[i][j] = (int *) malloc(dim3 * sizeof(int));
	    }
 
    }
    return array;
}

/** @function main */
int main( int argc, const char** argv ) {
	string image_n = argv[1];

	Mat frame = imread("source_images/dart"+image_n+".jpg", CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	
	// 3. Convert and blur input
	Mat img_gray, img_blur;
 	cvtColor( frame, img_gray, CV_BGR2GRAY );

	gaussian(img_gray, 7);
	imwrite("detected_darts/"+image_n+"/blurred.jpg", img_gray);

	Mat img_x(frame.size(), CV_32FC1);
	Mat img_y(frame.size(), CV_32FC1);
	Mat img_m(frame.size(), CV_32FC1);
	Mat img_d(frame.size(), CV_32FC1);

	// 4. Perform sobel edge detector and write outputs
	sobel(img_gray, img_x, img_y, img_m, img_d);
	write_sobel(img_x, img_y, image_n);

	Mat image_m = normalize_and_save(img_m, image_n, "magnitude");
	Mat image_d = normalize_and_save(img_d, image_n, "direction");

	// 5. Threshold magnitude, set threshold (between 0 and 255) for the normalised magnitude image
	threshold(image_m, threshold_magnitude, image_n, "source");

	// 6. Perform HoughTransform
	vector<vector<int> > circles = ch_transform(image_m, 20, min(frame.rows,frame.cols)/2, img_d, image_n);
	lh_transform(image_m, img_d, image_n);
	
	// 7. Detect Faces and Display Result
	detectAndDisplay( frame, read_csv(image_n), image_n, circles );

	// 8. Save Result Image
	imwrite( "detected_darts/"+image_n+"/detected_filtered.jpg", frame );

	return 0;
}

// Taken from COMS30020 Computer Graphics codebaes
std::vector<std::string> split(const std::string &line, char delimiter) {
	auto haystack = line;
	std::vector<std::string> tokens;
	size_t pos;
	while ((pos = haystack.find(delimiter)) != std::string::npos) {
		tokens.push_back(haystack.substr(0, pos));
		haystack.erase(0, pos + 1);
	}
	// Push the remaining chars onto the vector
	tokens.push_back(haystack);
	return tokens;
}

vector<Rect> read_csv(string num) {

	vector<Rect> truths;

	string file_name = "dart_truths/"+num+".csv";
	ifstream file(file_name);
	string line;

	while(getline(file, line)) {
		vector<string> tokens = split(line, ',');
		truths.push_back(Rect(stoi(tokens[0]),stoi(tokens[1]),stoi(tokens[2]),stoi(tokens[3])));
	}
	file.close();

	return truths;
}

void sobel(Mat &input, Mat &output_x, Mat &output_y, Mat &output_mag, Mat &output_dir) {

	Mat kX = Mat::ones(3, 3, CV_32F);
	
	// creating the sobel kernel for x
	kX.at<float>(0,0) = -1;
	kX.at<float>(1,0) = -2;
	kX.at<float>(0,1) = 0;
	kX.at<float>(1,1) = 0;
	kX.at<float>(1,2) = 2;
	kX.at<float>(1,2) = 2;
	kX.at<float>(2,0) = -1;
	kX.at<float>(2,1) = 0;

	// sobel kernel for y
	Mat kY = kX.t();

	int kernelRadiusX = ( kX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kX.size[1] - 1 ) / 2;

	Mat paddedInput;
	copyMakeBorder( input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, BORDER_REPLICATE );

	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			float sum_x = 0.0;
			float sum_y = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					float imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					float kernel_x = kX.at<float>( kernelx, kernely );
					float kernel_y = kY.at<float>( kernelx, kernely );

					sum_x += imageval * kernel_x;
					sum_y += imageval * kernel_y;
				}
			}
			output_x.at<float>(i, j) = (float) sum_x;
			output_y.at<float>(i, j) = (float) sum_y;
			output_mag.at<float>(i, j) = (float) sqrt((sum_y*sum_y) + (sum_x*sum_x));
			output_dir.at<float>(i, j) = (float) atan2(sum_y, sum_x);
		}
	}
}

void write_sobel(Mat x, Mat y, string num) {

	Mat r_img_x(x.size(), CV_8UC1);
	Mat r_img_y(x.size(), CV_8UC1);
	normalize(x,r_img_x,0,255,NORM_MINMAX, CV_8UC1);
    normalize(y,r_img_y,0,255,NORM_MINMAX, CV_8UC1);
    imwrite("detected_darts/"+num+"/x.jpg",r_img_x);
    imwrite("detected_darts/"+num+"/y.jpg",r_img_y);

}

Mat normalize_and_save(Mat input, string num, string name) {
	Mat n_input(input.size(), CV_8UC1, Scalar(0));
	normalize(input, n_input, 0, 255, NORM_MINMAX, CV_8UC1);

	imwrite("detected_darts/"+num+"/"+name+".jpg", n_input);

	return n_input;
}

void gaussian(Mat &input, int size) {

	Mat output(input.size(), input.type());

	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	cv::Mat kernel = kX * kY.t();

	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE );

	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;
					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );
					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			output.at<uchar>(i, j) = (uchar) sum;
		}
	}
	input = output.clone();
}

void filter_non_max(Mat &input_mag, Mat &input_dir) {

	assert(input_mag.size() == input_dir.size() && input_mag.type() == input_dir.type());

	for(int i = 1; i < input_mag.rows-1; i++) {
		for(int j = 1; j < input_mag.cols-1; j++) {
			double angle;
			if(input_dir.at<uchar>(i,j) >= 0) {
				angle = double(input_dir.at<uchar>(i,j));
			} else{
				angle = double(input_dir.at<uchar>(i,j)) + pi;
			}
			int r_angle = round(angle / (pi / 4));
			int mag = input_mag.at<uchar>(i,j);
			if((r_angle == 0 || r_angle == 4) && (input_mag.at<uchar>(i-1,j) > mag || input_mag.at<uchar>(i+1,j) > mag) || (r_angle == 1 && (input_mag.at<uchar>(i-1,j-1) > mag || input_mag.at<uchar>(i+1,j+1) > mag)) || (r_angle == 2 && (input_mag.at<uchar>(i,j-1) > mag || input_mag.at<uchar>(i,j+1) > mag)) || (r_angle == 3 && (input_mag.at<uchar>(i+1,j-1) > mag || input_mag.at<uchar>(i-1,j+1) > mag))) {
				input_mag.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
}

void threshold(Mat &input, int t, string num, string ver) {

	assert(t >= 0 && t <= 255);
	// output.create(input.size(), input.type());
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			int val = (int) input.at<uchar>(i, j);
			if(val > t) {
				input.at<uchar>(i,j) = (uchar) 255;
			} else {
				input.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
	imwrite("detected_darts/"+num+"/threshold_"+ver+".jpg", input);
}

void lh_transform(Mat &input, Mat &direction, string num) {

	assert(input.rows == direction.rows && input.cols == direction.cols);

	int diag = sqrt(pow(input.rows,2)+pow(input.cols,2));

	int **hough_space = malloc2dArray(diag,360);
    for (int i = 0; i < diag; i++) {
        for (int j = 0; j < 360; j++) {
            hough_space[i][j] = 0;
        }
    }
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			if(input.at<uchar>(x,y) == 255) {
				//for (int r = 0; r < r_max; r++) {
					int th = int(direction.at<float>(x,y)*(180/pi)) + 180;
					for(int t = th-5; t <= th+5; t++) {
						int mod_th = (t+360) % 360;
						float t_rad = (mod_th-180)*(pi/180);
						int xc = int(x * sin(t_rad));
						int yc = int(y * cos(t_rad));
						int p = xc + yc;
						if(p >= 0 && p <= diag) {
							hough_space[p][mod_th] += 1;
						}
					}
			}
        }
    }

	Mat hough_output(diag, 360, CV_32FC1, Scalar(0));
 
    for (int p = 0; p < diag; p++) {
        for (int t = 0; t < 360; t++) {
			hough_output.at<float>(p,t) = hough_space[p][t];
        }
    }

	Mat img_threshold = normalize_and_save(hough_output, num, "rho_theta_space");
	threshold(img_threshold, threshold_rho_space, num, "rho_theta");

	Mat hough_output_o(input.rows, input.cols, CV_32FC1, Scalar(0));
 
	for(int p = 0; p < hough_output.rows; p++) {
		for(int th = 0; th < hough_output.cols; th++) {
			if(img_threshold.at<uchar>(p,th) == 255) {
				float t_rad = (th-180) * (pi/180);
				for(int x = 0; x < input.cols; x++) {
					int y = ((-cos(t_rad))/sin(t_rad))*x + (p/sin(t_rad));

					if(y >= 0 && y < input.rows) {
						hough_output_o.at<float>(y,x)++;
					}
				}
			}
		}
	}
	Mat img_threshold_o = normalize_and_save(hough_output_o, num, "hough_space_lines");
	threshold(img_threshold_o, threshold_lines, num, "lines");
}

vector<vector<int>> ch_transform(Mat &input, int r_min, int r_max, Mat &direction, string num) {

	assert(input.rows == direction.rows && input.cols == direction.cols);

	int ***hough_space = malloc3dArray(input.rows, input.cols, r_max);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            for (int r = 0; r < r_max; r++) {
                hough_space[i][j][r] = 0;
            }
        }
    }
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			if(input.at<uchar>(x,y) == 255) {
				for (int r = 0; r < r_max; r++) {
					int xc = int(r * sin(direction.at<float>(x,y)));
					int yc = int(r * cos(direction.at<float>(x,y)));

					int a = x - xc;
					int b = y - yc;
					int c = x + xc;
					int d = y + yc;
					if(a >= 0 && a < input.rows && b >= 0 && b < input.cols) {
						hough_space[a][b][r] += 1;
					}
					if(c >= 0 && c < input.rows && d >= 0 && d < input.cols) {
						hough_space[c][d][r] += 1;
					}
				}
			}
        }
    }

	Mat hough_output(input.rows, input.cols, CV_32FC1);
 
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
            for (int r = r_min; r < r_max; r++) {
                hough_output.at<float>(x,y) += hough_space[x][y][r];
            }
 
        }
    }

	Mat hough_norm = normalize_and_save(hough_output, num, "hough_space_circles");

	vector<vector<int> > circles;
	for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			bool test_pass = true;
			map<int, int> t_circles;
            for (int r = r_min; r < r_max; r++) {
				if(hough_space[x][y][r] > threshold_circles) {
					t_circles[r] = hough_space[x][y][r];
				}
            }
			int max_c = 0;
			int max_r = 0;
			// for(int i = 0; i < circles.size(); i++) {
			// 	vector<int> circle = circles[i];
			// 	int xc = circle[0];
			// 	int yc = circle[1];
			// 	int rc = circle[2];

			// 	if(!(pow((x-xc),2) + pow((y-yc),2) > pow(rc,2))) {
			// 		test_pass = false;
			// 	}
			// }
			for(map<int, int>::const_iterator it = t_circles.begin(); it != t_circles.end(); ++it) {
				// if(it->second > max_c) {
				// 	max_r = it->first;
				// 	max_c = it->second;
				// }
				for(int i = 0; i < circles.size(); i++) {
					vector<int> circle = circles[i];
					int r = circle[2];
					if(r - 5 < it->first && r+5 > it->first){
						test_pass = false;
					}
				}
				if(test_pass) {
					vector<int> circle;
					circle.push_back(x);
					circle.push_back(y);
					circle.push_back(it->first);
					// cout << "radius: " << it->first << endl;
					circles.push_back(circle);
				}
			}
			// if(hough_space[x][y][max_r] > threshold && test_pass) {
			// 	vector<int> circle;
			// 	circle.push_back(x);
			// 	circle.push_back(y);
			// 	circle.push_back(max_r);
			// 	circles.push_back(circle);
			// }
        }
    }

	// cout << "circles: " << circles.size() << endl;

	return circles;
}

void draw_circles(Mat &input, vector<vector<int> > circles, string num) {
	Mat output = input.clone();
	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[1], c[0]);
		circle(output, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		int radius = c[2];
		circle(output, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}
	string xd = to_string(circles.size());
	imwrite("detected_darts/"+num+"/detected_circles_"+xd+".jpg", output);

}

//https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
float get_iou(Rect t, Rect d) {
	float width = min(d.x + d.width, t.x + t.width) - max(d.x, t.x);
	float height = min(d.y + d.height, t.y + t.height) - max(d.y, t.y);

	if(width <= 0 or height <= 0) return 0;

	float int_area = width * height;
	float uni_area = (d.width * d.height) + (t.width * t.height) - int_area;

	return int_area/uni_area;
}

//https://en.wikipedia.org/wiki/F-score
float get_f1_score(float t_p, float f_p, float f_n) {
	return (t_p == 0 && f_p == 0 && f_n == 0) ? 0 : t_p/(t_p + 0.5 * (f_p+f_n));
}

float average_radius(Mat &frame, vector<vector<int>> circles) {
	float avg = 0;
	for(auto c: circles) {
		avg += c[2];
	}
	return (circles.size() == 0) ? min(frame.cols,frame.rows)/4 : avg/circles.size();
}

int pixel_count(string num) {
	Mat load_lines = imread("detected_darts/"+num+"/threshold_lines.jpg", 1);
	Mat lines;
	cvtColor( load_lines, lines, CV_BGR2GRAY );
	int count = 0;
	for(int x = 0; x < lines.cols; x++) {
		for(int y = 0; y < lines.rows; y++) {
			if(lines.at<uchar>(y,x) == 255) count++;
		}
	}
	// cout << count << endl;
	return count;
}

vector<Rect> filter_darts(vector<Rect> darts, vector<vector<int> > circles, string num, int pixels) {
	vector<Rect> darts_filtered;
	Mat load_lines = imread("detected_darts/"+num+"/threshold_lines.jpg", 1);
	Mat lines;
	cvtColor( load_lines, lines, CV_BGR2GRAY );
	int lines_d = 0;
	for(int i = 0; i < darts.size(); i++) {
		int c_in = 0;
		int l_in = 0;
		for(int c = 0; c < circles.size(); c++) {
			int c_x = circles[c][1], c_y = circles[c][0];
			if(c_x > darts[i].x && c_x < darts[i].x+darts[i].width && c_y > darts[i].y && c_y < darts[i].y+darts[i].height) { c_in++; }
		}
		for(int x = darts[i].x; x < darts[i].x+darts[i].width; x++ ) {
			for(int y = darts[i].y; y < darts[i].y+darts[i].height; y++) {
				if(lines.at<uchar>(y,x) == 255) { l_in++; }
			}
		}
		// if(l_in != 0) cout << "rectangle " << darts[i] << " detected white pixels" << endl;
		int circle_threshold = (circles.size() > 0) ? (circles.size() + 6 - 1) / 6 : 1;
		int line_threshold = (pixels > 0) ? pixels / 8 : 1;
		// cout << "lines :" << l_in << "," << line_threshold << endl;
		// cout << "circle:" << c_in << "," << circle_threshold << endl;
		if(c_in >= circle_threshold || l_in >= line_threshold) {
			darts_filtered.push_back(darts[i]);
			lines_d++;
		}
	}
	// cout << "lines: " << lines_d << endl;
	return darts_filtered;
}

vector<Rect> filter_by_radius(vector<Rect> darts, float avg) {
	vector<Rect> filtered;
	for(auto d: darts) {
		if(min(d.width, d.height) < 4 * avg && min(d.width,d.height) > avg / 2) {
			filtered.push_back(d);
		}
	}
	return filtered;
}

vector<Rect> group_darts(vector<Rect> darts) {
	if(darts.size() == 0) return darts;
	vector<Rect> darts_grouped = { darts[0] };
	// vector<Rect> darts_grouped;
	darts.erase(darts.begin()+0);
	int i = 0;
	while(darts.size() > 0) {
		bool intersect = false;
		for(int j = 0; j < darts_grouped.size(); j++) {
			Rect avg = darts_grouped[j];
			if(get_iou(darts[0], avg) > 0 ) {
				darts_grouped[j].width = (avg.width + darts[0].width) / 2;
				darts_grouped[j].height = (avg.height + darts[0].height) / 2;
				darts_grouped[j].x = (avg.x + darts[0].x) / 2;
				darts_grouped[j].y = (avg.y + darts[0].y) / 2;
				intersect = true;
			} 
		}
		if(!intersect) darts_grouped.push_back(darts[0]);
		darts.erase(darts.begin()+0);
	}

	// int i = 0;
	// while(i < darts.size()) {
	// 	Rect avg = darts[i];
	// 	int j = 0;
	// 	while(j < darts.size()) {
	// 		if(get_iou(avg, darts[j]) > 0.0 && i != j) {
	// 			avg.width = (avg.width + darts[j].width) / 2;
	// 			avg.height = (avg.height + darts[j].height) / 2;
	// 			avg.x = (avg.x + darts[j].x) / 2;
	// 			avg.y = (avg.y + darts[j].y) / 2;
	// 			darts.erase(darts.begin()+j);
	// 		} else {
	// 			j++;
	// 		}
	// 	}
	// 	darts_grouped.push_back(avg);
	// 	darts.erase(darts.begin()+i);
	// 	i++;
	// }

	return darts_grouped;
}

void draw(Mat frame, vector<Rect> truths, vector<Rect> darts, vector<Rect> darts_f, vector<vector<int>> circles, string num) {
	Mat frame2 = frame.clone();
	// draw circles
	// draw truths
	for( int i = 0; i < truths.size(); i++) {
		rectangle(frame, Point(truths[i].x, truths[i].y), Point(truths[i].x + truths[i].width, truths[i].y + truths[i].height), Scalar( 0, 0, 255 ), 2);
		rectangle(frame2, Point(truths[i].x, truths[i].y), Point(truths[i].x + truths[i].width, truths[i].y + truths[i].height), Scalar( 0, 0, 255 ), 2);
	}
	for(int c = 0; c < circles.size(); c++) {
		circle(frame, Point(circles[c][1],circles[c][0]), 1, Scalar(0, 255, 255), 3, 8, 0);
		circle(frame2, Point(circles[c][1],circles[c][0]), 1, Scalar(0, 255, 255), 3, 8, 0);
	}
	draw_circles(frame, circles, num);
	// draw pre_filter
	for( int i = 0; i < darts.size(); i++ ) {
		rectangle(frame2, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
	}
	// draw post_filter
	for( int i = 0; i < darts_f.size(); i++ ) {
		rectangle(frame, Point(darts_f[i].x, darts_f[i].y), Point(darts_f[i].x + darts_f[i].width, darts_f[i].y + darts_f[i].height), Scalar( 0, 255, 0 ), 2);
	}
	// write file
	imwrite( "detected_darts/"+num+"/detected.jpg", frame2 );
}

void debug_out(vector<Rect> darts_filtered, vector<Rect> truths, string num) {
	
	float iou_threshold = 0.4;
	int true_darts = 0;

	for(int t = 0; t < truths.size(); t++) {
		for(int d = 0; d < darts_filtered.size(); d++) {
			if(get_iou(truths[t], darts_filtered[d]) > iou_threshold){
				// cout << truths[t] << endl << darts_filtered[d] << endl << get_iou(truths[t], darts_filtered[d]) << endl;
				true_darts++;
				break;
			}
		}
	}
	float tpr = (truths.size() > 0) ? (float)true_darts/(float)truths.size() : 0;
	float false_pos = darts_filtered.size() - true_darts;
	float false_neg = truths.size() - true_darts;
	float f1_score = get_f1_score(true_darts, false_pos, false_neg);

	cout << "image     : " << num << endl;
	cout << "tru darts : " << (float)truths.size() << endl;
	cout << "det darts : " << (float)darts_filtered.size() << endl;
	cout << "tpr       : " << (float)tpr << endl;
	cout << "false pos : " << (float)false_pos << endl;
	cout << "false neg : " << (float)false_neg << endl;
	cout << "f1 score  : " << (float)f1_score << endl << endl;

	// cout << tpr << endl;
	// for use with PyTorch for easy averaging
	// cout << "[" << num;
	// cout << "," << (float)truths.size();
	// cout << "," << (float)darts_filtered.size();
	// cout << "," << (float)tpr;
	// cout << "," << (float)false_pos;
	// cout << "," << (float)false_neg;
	// cout << "," << (float)f1_score << "]," << endl;
	// cout << "," << (float)(true_darts/(true_darts+false_pos)) << "]," << endl;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, vector<Rect> truths, string num, vector<vector<int> > circles ) {

	std::vector<Rect> darts;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	float avg = average_radius(frame, circles);

	// cout << "avg radius " << avg << endl;
	// cout << pixel_count(num) << endl;

	// 3. Filter bounding boxes based on hough transform
	vector<Rect> darts_filtered = filter_darts(darts, circles, num, pixel_count(num));

	// 3.1 Filter bounding boxes by radius (experimental)
	vector<Rect> darts_radius_filtered = filter_by_radius(darts_filtered, avg);

	// 4. Group filtered bounding boxes
	vector<Rect> darts_grouped = group_darts(darts_filtered);

	// 5. Draw boxes onto image
	draw(frame, truths, darts, darts_grouped, circles, num);

	// 6. Display TPR / IOU / F1
	debug_out(darts_grouped, truths, num);

}
