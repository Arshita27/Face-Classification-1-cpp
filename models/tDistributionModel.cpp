#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <iostream>
#include "boost/tuple/tuple.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
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
		Mat covarMat = Mat::eye(100, 100, CV_64FC1);
		covarMat = covarMat*(rand() % 255) * 100;

		return make_pair(meansMat.clone(), covarMat.clone());
		}


	pair <Mat, Mat> Estep(vector<vector<int> >dataset,
				Mat mean,
				Mat covar,
				double v,
				double D){
		cv::Mat expectation_h(dataset.size(), 1, CV_64FC1);
		cv::Mat expectation_logh(dataset.size(), 1, CV_64FC1);
		for (int i=0; i<dataset.size(); i++){
		    double arr[dataset[i].size()];
		    std::copy(dataset[i].begin(), dataset[i].end(), arr);
			cv::Mat single_dataset(1, 100, CV_64FC1, arr);
			Mat d = (single_dataset - mean)*(covar.inv()); // (1x100)
			Mat d1 = d*((single_dataset - mean).t());
			expectation_h.at<double>(i,0) = (v+D)/(v+d1.at<double>(0,0));
			expectation_logh.at<double>(i,0) = boost::math::digamma(((v/2) + (D/2)))
											- log((v/2)+ (d1.at<double>(i,0)/2));
		}
		return make_pair(expectation_h.clone(), expectation_logh.clone());
	}

	pair<Mat, Mat> Mstep(vector<vector<int> >dataset,
						Mat expectation_h,
						Mat expectation_logh,
						double v){

		vector<double> b;
		cv::reduce(expectation_h(Range::all(), Range::all()), b,0,CV_REDUCE_SUM);
		double y_sum = b.at(0);

		cv::Mat y_temp (dataset.size(), dataset.at(0).size(), CV_64FC1);
		for (int i=0; i<dataset.size(); i++){
			Mat temp=Mat(1,dataset.at(0).size(),CV_64FC1);
			for (int l = 0; l < dataset.at(i).size(); l++){
				temp.at<double>(0,l)= dataset[i][l];
			}
			y_temp.row(i) = expectation_h.at<double>(i,0)*temp.row(0);
		}
		cv::Mat new_mean (1, 100, CV_64FC1);
		for (int i=0; i<dataset.at(0).size(); i++){
			vector<double> a;
			cv::reduce(y_temp(Range::all(), Range(i,i+1)), a,0,CV_REDUCE_SUM);
			new_mean.at<double>(0,i) = a[0]/y_sum;
		}

		Mat covar_temp( dataset.size(), dataset.at(0).size()* dataset.at(0).size(), CV_64FC1);
		for (int i=0; i<dataset.size(); i++){
			double arr[dataset[i].size()];
			std::copy(dataset[i].begin(), dataset[i].end(), arr);
			cv::Mat single_dataset(1, 100, CV_64FC1, arr);
			cv::Mat d = ((single_dataset - new_mean).t())* (expectation_h.at<double>(i,0)*(single_dataset - new_mean));
			d.convertTo(d, CV_64FC1);
			d.reshape(1,1).row(0).copyTo(covar_temp.row(0));
		}
		cv::Mat new_covar(1, dataset.at(0).size()*dataset.at(0).size(), CV_64FC1);
		for (int i=0; i<dataset.at(0).size()*dataset.at(0).size(); i++){
			vector<double> a;
			cv::reduce(covar_temp(Range::all(), Range(i,i+1)), a,0,CV_REDUCE_SUM);
			new_covar.at<double>(0,i) = a[0]/y_sum;
		}
		new_covar = new_covar.reshape(1,100);

		return make_pair(new_mean.clone(), new_covar.clone());
	}

	double cal_v(double v, Mat expectation_logh, Mat expectation_h){
		return v;
	}

//	    def f(v):
//	        function_v = (t * ((v/2)* log(v/2))) + (t* log(scipy.special.gamma(v/2))) - (((v/2)-1)* np.sum(expectation_logh)) + ((v/2)*np.sum(expectation_h))
//	        return function_v
//	    final = scipy.optimize.fmin(f, v)
//	    new_v = final[0]


	void save_images(Mat mean_face,
					Mat covar_face,
					Mat mean_nonface,
					Mat covar_nonface){
		int check;
		char* SUB_DIR = "tDistributionResults";
		check = mkdir(SUB_DIR, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

		mean_face = mean_face.reshape(1,10);
		mean_nonface = mean_nonface.reshape(1,10);

		imwrite(SAVE_DIR + SUB_DIR + "/mean_face.jpg", mean_face);
		imwrite(SAVE_DIR + SUB_DIR + "/covar_face.jpg", covar_face);

		imwrite(SAVE_DIR + SUB_DIR + "/mean_nonface.jpg", mean_nonface);
		imwrite(SAVE_DIR + SUB_DIR + "/covar_nonface.jpg", covar_nonface);
	}

	Mat calNorm(vector<vector<int> >test,
						Mat mean,
						Mat covar,
						int v,
						int D){
		Mat pdf(test.size(), 1, CV_64FC1);
		for (int i=0; i<test.size(); i++){
			double arr[test[i].size()];
			std::copy(test[i].begin(), test[i].end(), arr);
			cv::Mat single_dataset(1, 100, CV_64FC1, arr);
			single_dataset = single_dataset.reshape(1,1);
			cv::Mat d = (single_dataset - mean)*(covar.inv());
			cv::Mat d1 =  d*(single_dataset - mean).t();
			pdf.at<double>(i,0) = (boost::math::lgamma(((v/2) + (D/2)) * pow((1+(d1.at<float>(0,0)/v)), (-(v+D)/2))))/(boost::math::lgamma(v/2));
			}
		return pdf;
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


	void train(vector<vector<int> > train_face,
				vector<vector<int> >train_nonface,
				vector<vector<int> >test_face,
				vector<vector<int> >test_nonface){

			pair<Mat, Mat> init_face = initialize_params();
			Mat mean_face = init_face.first;
			Mat covar_face = init_face.second;
			pair<Mat, Mat> init_nonface = initialize_params();
			Mat mean_nonface = init_nonface.first;
			Mat covar_nonface = init_nonface.second;
			double v_face = 300;
			double D_face = 100;
			double v_nonface = 300;
			double D_nonface = 100;
			Mat expectation_h_face;
			Mat expectation_logh_face;
			Mat expectation_h_nonface;
			Mat expectation_logh_nonface;

			std::cout<<"Parameters initialized"<<endl;
			for (int i=0; i<n_iterations; i++){
				std::cout<<"\niteration no: "<<i<<endl;

				pair<Mat, Mat> estep_face = Estep(train_face,
												mean_face,
												covar_face,
												v_face,
												D_face);
				expectation_h_face = estep_face.first;
				expectation_logh_face = estep_face.second;
				pair<Mat, Mat> mstep_face = Mstep(train_face,
												expectation_h_face,
												expectation_logh_face,
												v_face);
				std::cout<<"E-step complete for Face "<<endl;
				mean_face = mstep_face.first;
				covar_face = mstep_face.second;
				v_face= cal_v(v_face,
							expectation_h_face,
							expectation_logh_face);
				std::cout<<"M-step complete for Face "<<endl;
				pair<Mat, Mat> estep_nonface = Estep(train_nonface,
													mean_nonface,
													covar_nonface,
													v_nonface,
													D_nonface);
				expectation_h_nonface = estep_nonface.first;
				expectation_logh_nonface = estep_nonface.second;
				std::cout<<"E-step complete for Non-Face "<<endl;
				pair<Mat, Mat> mstep_nonface = Mstep(train_nonface,
												expectation_h_nonface,
												expectation_logh_nonface,
												v_nonface);
				mean_nonface = mstep_nonface.first;
				covar_nonface = mstep_nonface.second;
				v_nonface = cal_v(v_nonface,
								expectation_h_nonface,
								expectation_logh_nonface);
				std::cout<<"M-step complete for Non-Face "<<endl;
			}
			std::cout<<"\nStart Writing Mean Face and Mean NonFace Images"<<endl;
			save_images(mean_face, covar_face, mean_nonface, covar_nonface);
			std::cout<<"\nAll 'mean' Images written"<<endl;

			Mat logpdf_face_wrt_face, logpdf_face_wrt_nonface, logpdf_nonface_wrt_face, logpdf_nonface_wrt_nonface ;
			logpdf_face_wrt_face = calNorm(test_face, mean_face, covar_face, v_face, D_face);
			logpdf_face_wrt_nonface = calNorm(test_face, mean_nonface, covar_nonface, v_nonface, D_nonface);
			logpdf_nonface_wrt_face = calNorm(test_nonface, mean_face, covar_face, v_face, D_face);
			logpdf_nonface_wrt_nonface = calNorm(test_nonface, mean_nonface, covar_nonface, v_nonface, D_nonface);

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
