#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <iostream>
#include "boost/tuple/tuple.hpp"
#include <string>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace cv;
using namespace std;

class tDistribution{

private:
	string SAVE_DIR;
	int n_iterations;
public:
	tDistribution(string SAVE_DIR, int N_iterations):
		 SAVE_DIR(SAVE_DIR), n_iterations(N_iterations){}

	pair<Mat, Mat> initialize_params()
	{
		// Initial mean
		Mat meansMat(1, 100, CV_64FC1);
		double low = 0;
		double high = 255;
		randu(meansMat, Scalar(low), Scalar(high));

		// Initial covar
		vector <Mat> ch;
		for (int i=0; i<n_clusters; i++){
			Mat covar_cluster = Mat::eye(100, 100, CV_64FC1);
			covar_cluster = covar_cluster*(rand() % 255) * 1000;
			ch.push_back(covar_cluster);
		}
		Mat covarMat;
		merge(ch,covarMat);

		return make_pair(meansMat.clone(), covarMat.clone());
		}

	void train(vector<vector<int> > train_face,
				vector<vector<int> >train_nonface,
				vector<vector<int> >test_face,
				vector<vector<int> >test_nonface){

			clock_t ti;

			pair<Mat, Mat> init_face = initialize_params();
			Mat mean_face = init_face.first;
			Mat covar_face = init_face.second;
			pair<Mat, Mat> init_nonface = initialize_params();
			Mat mean_nonface = init_nonface.first;
			Mat covar_nonface = init_nonface.second;
			double v = 300;
			double D = 100;
			std::cout<<"Parameters initialized"<<endl;
			for (int i=0; i<n_iterations; i++){
				std::cout<<"\niteration no: "<<i<<endl;
			}
	}
};
