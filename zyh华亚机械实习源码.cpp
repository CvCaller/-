#include"head.h"

double calculatfunc::getDistance(Point point1, Point point2)
{
	double distance;
	distance = pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2);
	distance = sqrt(distance);
	return distance;
}
bool calculatfunc::comparePointByx(Point pt1, Point pt2)
{
	return pt1.x <= pt2.x;
}

void  Knife_detect_demo::KnifeImange_init(Mat& image)
{
	Knife_detect_demo kf;
	this->image = image;
	Mat gray, blur, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	//GaussianBlur(gray, blur, Size(3, 3), 0);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, se);
	imshow("binary", binary);
	waitKey(0);
	vector<vector<Point>>contours;
	vector<Vec4i>hier;
	findContours(binary, contours, hier, RETR_LIST, CHAIN_APPROX_SIMPLE);

	//vector<Rect>Knife_rects;
	for (int i = 0;i < contours.size();i++)
	{
		Rect rect = boundingRect(contours[i]);
		double area = contourArea(contours[i]);
		if (rect.height > binary.rows / 2)continue;
		if (area < 150)continue;
		boxes.push_back(rect);
	}
	this->binary = binary;
	//kf.Knife_sort(Knife_rects);
    detector = binary(boxes[1]);
	vector<Rect>detect;
	vector<int>nums;

	
}
void Knife_detect_demo::Knife_sort()
{
	for (int i = 0;i < boxes.size() - 1;i++)
	{
		for (int j = i;j < boxes.size();j++)
		{
			if (boxes[i].y > boxes[j].y)
			{
				Rect temp = boxes[i];
				boxes[i] = boxes[j];
				boxes[j] = temp;
			}
		}
	}
}
void Knife_detect_demo::detecting()
{
	vector<Mat>results;
	for (int i = 0;i < boxes.size();i++)
	{
		Rect roi = boxes[i];
		Mat obj = binary(roi);
		resize(obj, obj, detector.size());
		
		Mat result;
		subtract(detector, obj, result);
		Mat se = getStructuringElement(MORPH_RECT, Size(3,3), Point(-1, -1));
		morphologyEx(result, result, MORPH_OPEN, se);
		threshold(result, result, 0, 255, THRESH_BINARY);
		results.push_back(result);
		imshow("result", result);
		waitKey(0);
	}
	
	for (int i = 0;i < results.size();i++)
	{
		int score = 0;
		for (int h = 0;h < results[i].rows;h++)
		{
			for (int w = 0;w < results[i].cols;w++)
			{
				int data = results[i].at<uchar>(h, w);
				if (data == 255)score++;
			}
		}
		cout << score << endl;
		vector<vector<Point>>contours1;
		vector<Vec4i>hier;
		findContours(results[i], contours1, hier, RETR_LIST, CHAIN_APPROX_SIMPLE);
		bool find = false;
		for (size_t t = 0;t < contours1.size();t++)
		{
			Rect rect = boundingRect(contours1[t]);
			float radio = (float)(rect.width) / (float)(rect.height);
			int thresh = results[i].rows - (rect.y + rect.height);
			if (radio > 4.0 && (rect.y < 5 || thresh < 10))continue;
			double area = contourArea(contours1[t]);
			if (area > 10)find = true;
		}
		cout << find << endl;
		if (score > 50 && find)
		{
			detect.push_back(boxes[i]);
			nums.push_back(i);
		}
	
	}
	//cout << detect.size() << endl;
	for (int i = 0;i < detect.size();i++)
	{
		rectangle(this->image, detect[i], Scalar(0, 0, 255), 2, 8);
	}
	for (int i = 0;i < nums.size();i++)
	{
		cout << nums[i] << endl;
	}
	imshow("image", image);
	waitKey(0);
}


void HOG_SVM_detect::get_hog_descriptor(Mat &image, vector<float>& desc)
{
	
	HOGDescriptor hog;
	Mat img, gray;
	int h = image.rows;
	float rate = 64.0 / image.cols;
	resize(image, img, Size(64, int(rate * h)));
	cvtColor(img, gray, COLOR_BGR2GRAY);
	Mat winrect = Mat::zeros(Size(64, 128), CV_8UC1);
	winrect = Scalar(127);
	Rect roi;
	roi.x = 0;
	roi.y = (128 - gray.rows) / 2;
	roi.width = 64;
	roi.height = gray.rows;
	gray.copyTo(winrect(roi));
	hog.compute(winrect, desc, Size(8, 8), Size(0, 0));
}
void HOG_SVM_detect::generate_dataset()
{
	vector<string>postive_image;
	glob("D:/资料图像/elec_watch/positive", postive_image);
	for (int i = 0;i < postive_image.size();i++)
	{
		cout << postive_image[i] << endl;
	}
	
	for (int k = 0;k < postive_image.size();k++)
	{
		Mat detection = imread(postive_image[k].c_str());
		vector<float>postive_desc;
		get_hog_descriptor(detection, postive_desc);
		for (int j = 0;j <postive_desc.size();j++)
		{
			trainData.at<float>(k, j) = postive_desc[j];
		}
		labels.at<int>(k, 0) = 1;
	}
	vector<string>negative_image;
	glob("D:/资料图像/elec_watch/negative", negative_image);
	
	for (int a = 0;a < negative_image.size();a++)
	{
		Mat detection = imread(negative_image[a].c_str());
		vector<float>negative_desc;
		get_hog_descriptor(detection, negative_desc);
		for (int b = 0;b < negative_desc.size();b++)
		{
			trainData.at<float>(a+ postive_image.size(), b) = negative_desc[b];
		}
		labels.at<int>(a + postive_image.size(), 0) = -1;
	}
	//cout << trainData << endl;
	
}
void HOG_SVM_detect::svm_train()
{
	printf("\n start SVM training... \n");
	svm->setC(2.67);
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(5.383);
	svm->train(trainData, ROW_SAMPLE, labels);
	printf("end train...\n");

	svm->save("D:/资料图像/elec_watch/self_hog_elec.xml");
}
void HOG_SVM_detect::detecting(Mat frame)
{
	//Mat image = frame.clone();

	Ptr<SVM> svm1 = SVM::load("D:/资料图像/elec_watch/self_hog_elec.xml");
	resize(frame, frame, Size(0, 0), 0.2, 0.2);
	//frame.convertTo(frame, CV_8SC3);
	Rect roi;
	roi.height = 128;
	roi.width = 64;
	int sum_x = 0;
	int sum_y = 0;
	int cont = 0;
	for (int h = 64;h < frame.rows-64;h+=4)
	{
		
		for (int w = 32;w < frame.cols - 32;w+=4)
		{
			roi.y = h - 64;
			roi.x = w - 32;
			vector<float>desc;
			Mat image = frame(roi);
			this->get_hog_descriptor(image, desc);
			Mat fv = Mat::zeros(Size(desc.size(),1 ), CV_32FC1);
			for (int i = 0;i < desc.size();i++)
			{
				fv.at<float>(0, i) = desc[i];
			}
			float result=svm1->predict(fv);
			if (result > 0) 
			{
				cont += 1;
				sum_x += roi.x;
				sum_y += roi.y;
				
				//rectangle(frame, roi, Scalar(0,0,255), 1, 8);
			}
		}
	}
	if (cont == 0)
	{
		imshow("frame", frame);
		waitKey(0);
		return;
	}
	roi.x = sum_x / cont;
	roi.y = sum_y / cont;
	rectangle(frame, roi, Scalar(0, 0, 255), 1, 8);
	imshow("frame", frame);
	waitKey(0);
}



Mat QRcodedetect::transform(Mat& image)
{
	Mat image1 = image.clone();
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	vector<vector<Point>>contours;
	findContours(binary, contours,	RETR_LIST, CHAIN_APPROX_SIMPLE, Point());
	Mat result1 = Mat::zeros(image.size(), CV_8UC1);
	vector<vector<Point>>contours_pts;
	for (int i = 0;i < contours.size();i++)
	{
		double area = contourArea(contours[i]);
		if (area < 100)continue;
		RotatedRect rect = minAreaRect(contours[i]);
		float w = rect.size.width;
		float h = rect.size.height;
		float rate = min(w, h) / max(w, h);
		
		if (rate > 0.85 && w < binary.cols / 4 && h < binary.rows / 4)
		{
			int width = static_cast<int>(rect.size.width);
			int height = static_cast<int>(rect.size.height);
			Mat result2 = Mat::zeros(height, width, image.type());
			Point2f ver[4];
			rect.points(ver);
			vector<Point>src_pts, dst_pts;
			dst_pts.push_back(Point(0, 0));
			dst_pts.push_back(Point(width, 0));
			dst_pts.push_back(Point(width, height));
			dst_pts.push_back(Point(0, height));
			for (int j=0;j < 4;j++)src_pts.push_back(ver[j]);
			Mat h = findHomography(src_pts, dst_pts);
			warpPerspective(image, result2, h, result2.size());
			if (ISxcorner(result2,i))
			{
				//imshow("result" + to_string(i), result2);
				contours_pts.push_back(contours[i]);
			}

		}		
	}
	vector<vector<Point>>QRcodeCorner;
	QRcodeCornerdetect(QRcodeCorner, contours_pts);//过滤误判轮廓
	
		for (int i = 0;i < QRcodeCorner.size();i++)
		{
			drawContours(result1, QRcodeCorner, i, Scalar(255, 255, 255), 2, 8);
		}
		vector<Point>pts;
		for (int h = 0;h < result1.rows;h++)
		{
			for (int w = 0;w < result1.cols;w++)
			{
				int pv = result1.at<uchar>(h, w);
				if (pv == 255)
				pts.push_back(Point(w, h));
			}
		}
		vector<Point>src_pts;
		RotatedRect rrt = minAreaRect(pts);
		Point2f temp[4];
		rrt.points(temp);
		for (int i = 0;i < 4;i++)
		{
			cout << temp[i] << endl;
			line(image, temp[i], temp[(i + 1) % 4], Scalar(0, 0, 255), 2, 8);
			src_pts.push_back(temp[i]);
		}
		Mat result3 = Mat::zeros(Size(200, 200), image.type());
		vector<Point> dst_pts;
		dst_pts.push_back(Point(0,0));
		dst_pts.push_back(Point(200, 0));
		dst_pts.push_back(Point(200, 200));
		dst_pts.push_back(Point(0, 200));
		Mat h = findHomography(src_pts, dst_pts);
		warpPerspective(image, result3, h, result3.size());

		imwrite("D:/资料图像/test.png", result3);
		imwrite("D:/资料图像/test1.png", image);
		imshow("result", result3);
		return image;
}
	

bool QRcodedetect::ISxcorner(Mat binary,int t)
{
	cvtColor(binary, binary, COLOR_BGR2GRAY);
	threshold(binary, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow(to_string(t), binary);
	int w = binary.cols;
	int h = binary.rows;
	int b1x = 0; int w1x = 0; int b2x = 0; int w2x = 0;
	Point center = Point(w / 2, h / 2);
	int pv = binary.at<int>(h / 2, w / 2);
	if (pv==255)return false;
	int start = 0;int end = 0;
	int offex = 0;
	bool findleft = false;
	bool findright = false;
	while (true)
	{
		offex++;
		pv = binary.at<uchar>(h / 2, (w / 2 - offex));
		if (pv==255)
		{
			start = w / 2 - offex;
			findleft = true;
		}
		pv = binary.at<uchar>(h / 2, (w / 2 + offex));
		if (pv==255)
		{
			end = w / 2 + offex;
			findright = true;
		}
		if (findleft && findright)break;
		if ((w / 2 - offex) < w / 8 || (w / 2 + offex) > w - 1)
		{
			start = -1;
			end = -1;
			break;
		}
	}
	if (start < 0 || end < 0)return false;
	int xb = end - start;
	for (int i = start;i > 0;i--) 
	{
		pv = binary.at<uchar>(h / 2, i);
		if (pv == 0)break;
		w1x++;
	}
	for (int j = end;j<w - 1;j++)
	{
		pv = binary.at<uchar>(h / 2, j);
		if (pv == 0)break;
		w2x++;
	}
	
	for (int k = start - w1x;k > 0;k--)
	{
		pv = binary.at<uchar>(h / 2, k);
		if (pv == 0)b1x++;
	}
	for (int l = end + w2x;l < w;l++)
	{
		pv = binary.at<uchar>(h / 2, l);
		if (pv == 0)b2x++;
	}
	cout << "序号：" << t << endl;
	printf("xb : %d, b1x = %d, b2x = %d, w1x = %d, w2x = %d\n", xb, b1x, b2x, w1x, w2x);
	float sum = xb + w1x + w2x + b1x + b2x;
	xb = static_cast<int>((xb / sum) * 7.0 + 0.5);
	w1x = static_cast<int>((w1x / sum) * 7.0 + 0.5);
	w2x = static_cast<int>((w2x / sum) * 7.0 + 0.5);
	b1x = static_cast<int>((b1x / sum) * 7.0 + 0.5);
	b2x = static_cast<int>((b2x / sum) * 7.0 + 0.5);
	printf("xb : %d, b1x = %d, b2x = %d, w1x = %d, w2x = %d\n", xb, b1x, b2x, w1x, w2x);
	cout << "sum:" << sum << endl;
	if ((xb == 3 || xb == 4) && w1x==w2x&&w2x==b1x&&b1x==b2x&&b2x==1)
	{
		return true;
	}
	else
	{
		return false;
	}
	
}
bool QRcodedetect::ISycorner(Mat binary, int t)
{
	cvtColor(binary, binary, COLOR_BGR2GRAY);
	threshold(binary, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	int w = binary.cols;
	int h = binary.rows;
	int pv = binary.at<uchar>(h / 2, w / 2);
	if (pv == 255)return false;
	int yb = 0;int w1y = 0; int w2y = 0; int b1y = 0;int b2y = 0;
	int offex = 0;int start = 0;int end = 0;
	bool findup = false;
	bool finddown = false;
	while (true)
	{
		offex++;
		pv = binary.at<uchar>(h / 2 - offex, w / 2);
		if (pv == 255)
		{
			start = h / 2 - offex;
			findup = true;
		}
		pv = binary.at<uchar>(h / 2 + offex, w / 2);
		if (pv == 255)
		{
			end = h / 2 + offex;
			finddown = true;
		}
		if (finddown && findup)break;
		if ((h / 2 - offex) < h / 8 || (h / 2 + offex) > h - 1)
		{
			start = -1;
			end = -1;
			break;
		}
	}
	if (start < 0 || end < 0)return false;
	yb = end - start;
	for (int i = 0;i < start;i++)
	{
		pv = binary.at<uchar>(start - i, w / 2);
		if (pv == 0)break;
		w1y++;
	}
	for (int i = end;i < h-1;i++)
	{
		pv = binary.at<uchar>(i, w / 2);
		if (pv == 0)break;
		w2y++;
	}
	for (int i = (start - w1y);i >0;i--)
	{
		pv = binary.at<uchar>(i, w / 2);
		if (pv == 255)break;
		b1y++;
	}
	for (int i = (end+ w2y);i <h;i++)
	{
		pv = binary.at<uchar>(i, w / 2);
		if (pv == 255)break;
		b2y++;
	}
	cout << "序号：" << t << endl;
	cout << "yb:" << yb << endl;
	cout << "w1y:" << w1y << endl;
	cout << "w2y:" << w2y << endl;
	cout << "b1y:" << b1y << endl;
	cout << "b2y:" << b2y << endl;
	float sum = yb + w1y + w2y + b1y + b2y;
	yb = static_cast<int>((yb / sum) * 7.0 + 0.5);
	w1y = static_cast<int>((w1y / sum) * 7.0 + 0.5);
	w2y = static_cast<int>((w2y / sum) * 7.0 + 0.5);
	b1y = static_cast<int>((b1y / sum) * 7.0 + 0.5);
	b2y = static_cast<int>((b2y / sum) * 7.0 + 0.5);
	
	cout << "yb:" << yb << endl;
	cout << "w1y:" << w1y << endl;
	cout << "w2y:" << w2y << endl;
	cout << "b1y:" << b1y << endl;
	cout << "b2y:" << b2y << endl;
	cout << "sum:" << sum << endl;
	if ((yb == 3 || yb == 4) && w1y == w2y && w2y == b1y && b1y == b2y&&b2y==1)
	{
		return true;
	}
	else 
	{
		return false;
	}
}
void QRcodedetect::QRcodeCornerdetect(vector<vector<Point>>&QRcodeCorner, vector<vector<Point>>contours_pts)
{
	for (int i = 0;i < contours_pts.size();i++)
	{
		RotatedRect r0 = minAreaRect(contours_pts[i]);
		Point pt0 = r0.center;
		double distance1 = 0.0;
		double distance2 = 0.0;
		for (int k = 0;k < contours_pts.size();k++)
		{
			RotatedRect r1 = minAreaRect(contours_pts[k]);
			Point pt1 = r1.center;
			float w = r0.boundingRect().width;
			distance1 = calculatfunc::getDistance(pt0, pt1);
			for (int j = 0;j < contours_pts.size();j++)
			{
				RotatedRect r2 = minAreaRect(contours_pts[j]);
				Point pt2 = r2.center;
				distance2 = calculatfunc::getDistance(pt0, pt2);
				double rate1 = min(distance1, distance2) / max(distance1, distance2);
				float rate2 = distance1 / w;

				if (rate1 > 0.9 && j != k && rate2 < 4)
				{
					QRcodeCorner.push_back(contours_pts[i]);
					QRcodeCorner.push_back(contours_pts[j]);
					QRcodeCorner.push_back(contours_pts[k]);
					return;
				}
			}

		}
	}
}

 void arrayDetect::arraydetecting(Mat image,int t)//t为检测阈值，0<t<5,t越小越严格
{
	 Mat gray, binary;
	 cvtColor(image, gray, COLOR_BGR2GRAY);
	 threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	 Mat se = getStructuringElement(MORPH_RECT, Size(3, 3));
	 morphologyEx(binary, binary, MORPH_OPEN, se);
	 vector<vector<Point>>contours;
	 findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	 
     #pragma omp parallel for//OpenMP并行加速
	 for (int i = 0;i < contours.size();i++)
	 {
		 int area = contourArea(contours[i]);
		 if (area < 500)continue;
		 vector<int>left;
		 Rect rect = boundingRect(contours[i]);
		 for (int j = (rect.tl().y + 2);j < (rect.br().y - 2);j++)
		 {
			 for (int k = (rect.tl().x-5);k < (rect.tl().x+5);k++)
			 {
				 int pv = binary.at<uchar>(j, k);
				 if (pv>0)
				 {
					 left.push_back(k);
					 break;
				 }
			 }
		 }
		 int left_dx=0;
		 if (left.size() > 0)
		 {
			 sort(left.begin(), left.end());
			  left_dx = left[left.size() - 1] - left[0];
		 }
		 
		 vector<int>right;
         for (int j = (rect.tl().y + 2);j < (rect.br().y - 2);j++)
		 {
			 for (int k = (rect.br().x + 5);k > (rect.br().x - 5);k--)
			 {
				 int pv = binary.at<uchar>(j, k);
				 if (pv > 0)
				 {
					 right.push_back(k);
					 break;
				 }
			 }
		 }
		 int right_dx = 0;
		 if (right.size() > 0)
		 {
			 sort(right.begin(), right.end());
			 right_dx = right[right.size() - 1] - right[0];
		 }
         

		
		 if(left_dx>t||right_dx>t)
			 drawContours(image, contours, i, Scalar(0, 0, 255), 2, 8);
		 
		
	 }
	
	 imshow("image", image);
	 imwrite("D:/资料图像/test1.png", image);
	 waitKey(0);
	 

	 
	
	 


	
}

 ORBDetector::ORBDetector()
 {
	 cout << "create orb detector..." << endl;
 }
 ORBDetector::~ORBDetector()
 {
	 this->tpl.release();
	 this->tpl_desc.release();
	 this->tpl_kpts.clear();
	 this->detector.release();
	 cout << "release orb detector..." << endl;
 }
 void ORBDetector::initORB(Mat& refImage)
 {
	 if (!refImage.empty())
	 {
		 Mat tplGray;
		 cvtColor(refImage, tplGray, COLOR_BGR2GRAY);
		 detector->detectAndCompute(tplGray, Mat(), this->tpl_kpts, this->tpl_desc);
		 tplGray.copyTo(this->tpl);
	 }

 }
 bool ORBDetector::detect_and_analysis(Mat& image, Mat& aligned)
 {
	 Mat img2gray;
	 cvtColor(image, img2gray, COLOR_BGR2GRAY);
	 vector<KeyPoint>img_kpts;
	 Mat img_desc;
	 this->detector->detectAndCompute(img2gray, Mat(), img_kpts, img_desc);
	 vector<DMatch>matches;
	 //Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
	 Ptr<DescriptorMatcher>matcher = DescriptorMatcher::create("BruteForce-Hamming");//"BruteForce-Hamming"
	 matcher->match(img_desc, this->tpl_desc, matches, Mat());
	 float GOOD_MATCH_PERCENT = 0.15f;
	 const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	 sort(matches.begin(), matches.end());
	 bool found = true;
	 if (matches[0].distance > 30)found = false;
	 matches.erase(matches.begin() + numGoodMatches, matches.end());
	 vector<Point2f>obj_pts, scene_pts;
	 for (size_t i = 0; i < matches.size(); i++)
	 {
		 scene_pts.push_back(img_kpts[matches[i].queryIdx].pt);
		 obj_pts.push_back(this->tpl_kpts[matches[i].trainIdx].pt);
	 }
	 Mat h = findHomography(scene_pts, obj_pts, RANSAC);
	 Mat result;
	 warpPerspective(image, result, h, tpl.size());
	 rotate(result, result, ROTATE_90_COUNTERCLOCKWISE);
	 result.copyTo(aligned);
	 return found;
 }