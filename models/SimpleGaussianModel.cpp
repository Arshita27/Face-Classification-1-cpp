#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>


using namespace cv;
using namespace std;

class SimpleGaussian{
private:
	string SAVE_DIR;
public:
	SimpleGaussian(string SAVE_DIR): SAVE_DIR(SAVE_DIR) {};

	pair <Mat, Mat> initialize_param(vector<vector<int> > data)
	{
		float dmat[917][100]={0};
		for (int i=0; i<data.size(); i++){
			for (int j=0; j<data[0].size(); j++){
				dmat[i][j]=data[i][j];
			}
		}
		Mat dmatrix(917, 100, CV_32FC1, dmat);

		Mat covarMats;
		Mat means;
		calcCovarMatrix(dmatrix, covarMats, means,  CV_COVAR_NORMAL| CV_COVAR_ROWS);
		means = means.reshape(1,10);
		return make_pair(means, covarMats);
	}

	vector<float> calNorm(vector<vector<int> >test, Mat mean, Mat covar)
	{	mean = mean.reshape(1,100);
		covar.inv();
		vector<float>log_pdf;
		for (int i=0; i<test.size(); i++){
			float test_mat[1][100]={0};
			for (int j=0; j<test[0].size(); j++){
				test_mat[0][j]=test[i][j];
				}
			Mat singleTest(1, 100, CV_32FC1, test_mat);
			singleTest = singleTest.reshape(1,100);
			singleTest.convertTo(singleTest,  CV_32FC1);
			mean.convertTo(mean, CV_32FC1);
			covar.convertTo(covar, CV_32FC1);

			Mat d = (covar.inv())*(singleTest - mean);
			Mat d1 = ((singleTest - mean).t())*d;
			float pdf = exp(d1.at<float>(0,0)*(-0.5));//
			log_pdf.push_back(log(pdf));
		}
		return log_pdf;
	}

	pair< vector<float>, vector<float> > posterior(vector<float> log_pdf_face, vector<float> log_pdf_nonface)
	{
		vector<float> pos_face, pos_nonface;
		for (int i=0; i<log_pdf_face.size(); i++){
			pos_face.push_back(log_pdf_face[i]/(log_pdf_face[i] + log_pdf_nonface[i]));
			pos_nonface.push_back(log_pdf_nonface[i]/(log_pdf_face[i] + log_pdf_nonface[i]));
		}
		return make_pair(pos_face, pos_nonface);
	}

	void save_images(Mat mean_face, Mat covar_face, Mat mean_nonface, Mat covar_nonface){
		int check;
		char* SUB_DIR = "SimpleGaussianResults";
		check = mkdir(SUB_DIR, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

		imwrite(SAVE_DIR + SUB_DIR + "/mean_face.jpg", mean_face);
		imwrite(SAVE_DIR + SUB_DIR + "/covar_face.jpg", covar_face);

		imwrite(SAVE_DIR + SUB_DIR + "/mean_nonface.jpg", mean_nonface);
		imwrite(SAVE_DIR + SUB_DIR + "/covar_nonface.jpg", covar_nonface);
	}

	void train(vector<vector<int> > train_face,
			vector<vector<int> >train_nonface,
			vector<vector<int> >test_face,
			vector<vector<int> >test_nonface){

		pair<Mat, Mat> p_face = initialize_param(train_face);
		cv::Mat mean_face = p_face.first;
		cv::Mat covar_face = p_face.second;
		pair<Mat, Mat> p_nonface = initialize_param(train_nonface);
		cv::Mat mean_nonface = p_nonface.first;
		cv::Mat covar_nonface = p_nonface.second;

		save_images(mean_face, covar_face, mean_nonface, covar_nonface);

		vector<float>logpdf_face_wrt_face, logpdf_face_wrt_nonface, logpdf_nonface_wrt_face, logpdf_nonface_wrt_nonface ;
		logpdf_face_wrt_face = calNorm(test_face, mean_face, covar_face);
		logpdf_face_wrt_nonface = calNorm(test_face, mean_nonface, covar_nonface);
		logpdf_nonface_wrt_face = calNorm(test_nonface, mean_face, covar_face);
		logpdf_nonface_wrt_nonface = calNorm(test_nonface, mean_nonface, covar_nonface);

		vector<float> pos_testface_face, pos_testface_nonface, pos_testnonface_face, pos_testnonface_nonface;
		pair<vector<float>,vector<float> > pos_testface = posterior(logpdf_face_wrt_face, logpdf_face_wrt_nonface);
		pos_testface_face = pos_testface.first;
		pos_testface_nonface = pos_testface.second;

		pair<vector<float>,vector<float> > pos_testnonface = posterior(logpdf_nonface_wrt_face, logpdf_nonface_wrt_nonface);
		pos_testnonface_face = pos_testnonface.first;
		pos_testnonface_nonface = pos_testnonface.second;

		int count_false_positive = 0;
		for (int i =0; i<pos_testface_face.size(); i++){
			if (pos_testnonface_face.at(i)>0.5)
				count_false_positive+=1;
		}
		cout<<"False Positive Rate: "<< (count_false_positive)/(pos_testface_face.size())<<endl;

		int count_false_negative = 0;
		for (int i =0; i<pos_testface_face.size(); i++){
			if (pos_testnonface_nonface.at(i)>0.5)
				count_false_negative+=1;
		}
		cout<<"False Negative Rate: "<< (count_false_negative)/(pos_testface_face.size())<<endl;
	}
};
