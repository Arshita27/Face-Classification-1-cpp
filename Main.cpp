#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include "SimpleGaussianModel.cpp"
#include "MixtureGaussianModel.cpp"

using namespace cv;
using namespace std;


vector<vector<int> > get_images(String patern){
	vector<String> images;
	glob(patern, images);

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

int main(){
	clock_t t;

	vector<vector<int> > train_face, train_nonface, test_face, test_nonface;
	vector<String> patern;
	patern.push_back("/home/arshita/Desktop/Dataset/train_face/*.jpg");
	patern.push_back("/home/arshita/Desktop/Dataset/train_non/*.jpg");
	patern.push_back("/home/arshita/Desktop/Dataset/test_face/*.jpg");
	patern.push_back("/home/arshita/Desktop/Dataset/test_non/*.jpg");

	train_face = get_images(patern[0]);
	train_nonface = get_images(patern[1]);
	test_face = get_images(patern[2]);
	test_nonface = get_images(patern[3]);

	// Simple Gaussian
	SimpleGaussian sim_gaus;
	sim_gaus.train(train_face, train_nonface, test_face, test_nonface);

	// Mixture of Gaussian
//	int num_clusters = 3;
//	MixtureGaussian mix_gaus(num_clusters);
//	t = clock();
//	mix_gaus.train(train_face, train_nonface, test_face, test_nonface);
//	t = clock() - t;
//	cout<<"\nTotal time taken: "<<((float)t)/CLOCKS_PER_SEC<<" seconds"<<endl;

}

// NOTE:
// create config file and keep all root dir in it.



