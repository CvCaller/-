#include<opencv2/opencv.hpp>
#include<openvino/openvino.hpp>
#include<iostream>
#include <omp.h>
using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace ov;
class calculatfunc
{
public:
	static  double getDistance(Point point1, Point point2);
	static bool comparePointByx(Point pt1, Point pt2);

};
class Knife_detect_demo
{
private:
	
	Mat binary;
	vector<Rect> boxes;
	Mat detector;
	vector<Rect>detect;
	vector<int>nums;
	Mat image;
public:
	void KnifeImange_init(Mat &image);
	void Knife_sort();
	void detecting();
};

class HOG_SVM_detect
{
public:
	void detecting(Mat frame);
	void get_hog_descriptor(Mat& image, vector<float>& desc);
	void generate_dataset();
	void svm_train();

private:
	Mat trainData= Mat::zeros(Size(3780, 26), CV_32FC1);
	Mat  labels = Mat::zeros(Size(1, 26), CV_32SC1);
	Ptr<SVM> svm = SVM::create();
};

class QRcodedetect
{
public :
	Mat transform(Mat& image);
	bool ISxcorner(Mat binary,int t);
	bool ISycorner(Mat binary, int t);
	void QRcodeCornerdetect(vector<vector<Point>>&QRcodeCorner, vector<vector<Point>>contours_pts);
private:
	Mat temp;
};

class arrayDetect
{
public:
	static void arraydetecting(Mat image,int t);
    
};

class ORBDetector
{
private:
	Ptr<ORB> detector = ORB::create(500);
	Mat tpl;
	vector<KeyPoint>tpl_kpts;
	Mat tpl_desc;
public:
	ORBDetector(void);
	~ORBDetector(void);
	void initORB(Mat& refImage);
	bool detect_and_analysis(Mat& image, Mat& aligned);
};