#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <cstdlib>
#include <time.h>
#include <iostream>

using namespace std;
using namespace cv;

vector<vector<int> > get_images(string FULL_PATH){
	vector<String> images;
	glob(FULL_PATH, images);

	vector<vector<int> > data;
	for(int i=0;i<images.size();i++){
		cv::Mat img = imread(images[i], IMREAD_GRAYSCALE); // image loaded as gray scale
		img =img.reshape(1,1);
		if(!img.data) cout<<"Cant open"<<images[i]<<endl;
		vector<double> array;
		array.assign((uchar*)img.datastart, (uchar*)img.dataend);
		data.push_back(img);
	}
	return data;
}

vector<string> get_image_Path(String ROOT_DIR){
	vector<string> FULL_PATH;
	String SUB_DIR[4] =  {"train_face", "train_non", "test_face", "test_non"};
	for (int i=0; i<4; i++){
		FULL_PATH.push_back(ROOT_DIR + SUB_DIR[i] + "/*.jpg" );
	}
	return FULL_PATH;
}
