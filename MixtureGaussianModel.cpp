#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <iostream>
#include "boost/tuple/tuple.hpp"
#include <string>
#include <time.h>
using namespace cv;
using namespace std;


class MixtureGaussian{

private:
	int n_clusters;
public:
	MixtureGaussian(int N_clusters): n_clusters(N_clusters){}

	pair<Mat, Mat> initialize_params()
	{
		// Initial mean
		double means [n_clusters][100 ];
		for (int i=0; i<n_clusters; i++){
			for (int j=0; j<100; j++){
				means[i][j] = rand() % 255;
			}
		}
		Mat meansMat(n_clusters, 100, CV_64FC1, means);

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

	pair <Mat, Mat> e_step(vector<vector<int> >dataset, vector<double> lambda, Mat mean, Mat covar)
	{
		cv::Mat pdf(dataset.size(), n_clusters, CV_64FC1);
		cv::Mat sum_pdf(dataset.size(), 1, CV_64FC1);

		cv::Mat covar_cluster[n_clusters];
		split(covar, covar_cluster);

		for (int i=0; i<dataset.size(); i++){
			for (int j=0; j<n_clusters; j++){
			    double arr[dataset[i].size()];
			    std::copy(dataset[i].begin(), dataset[i].end(), arr);
				cv::Mat single_dataset(1, 100, CV_64FC1, arr);
				single_dataset = single_dataset.reshape(1,1);
				cv::Mat d = (single_dataset - mean.row(j))*(covar_cluster[j].inv());
				cv::Mat d1 =  d*(single_dataset - mean.row(j)).t();
				pdf.at<double>(i,j)= double(lambda[j])*double(exp(d1.at<double>(0,0)*(-0.5)));
			}
			vector<double> a;
			cv::reduce(pdf(Range(i,i+1), Range::all()), a,1,CV_REDUCE_SUM);
			sum_pdf.at<double>(i,0) = a.at(0);
		}
		cv::Mat gamma(dataset.size(), n_clusters, CV_64FC1);
		for (int i=0; i<gamma.rows; i++){
			for (int j=0; j<gamma.cols; j++){
				gamma.at<double>(i,j) = double(pdf.at<double>(i,j))/double(sum_pdf.at<double>(i,0));
			}
		}
		cv::Mat gamma_temp(n_clusters, 1, CV_64FC1);
		for (int i=0; i<n_clusters; i++){
			vector<double> b;
			cv::reduce(pdf(Range::all(), Range(i,i+1)), b,0,CV_REDUCE_SUM);
			gamma_temp.at<double>(i,0) = b.at(0);
		}
		return make_pair(gamma.clone(), gamma_temp.clone());
	}

	pair <Mat, Mat> m_step(vector<vector<int> >dataset, Mat gamma, Mat gamma_temp){

		// ****************** new lambda **********************
		vector<double> sum_gamma_temp;
		cv::reduce(gamma_temp.col(0), sum_gamma_temp,0,CV_REDUCE_SUM);
////		vector<double> new_lambda;
//		for (int i =0; i<n_clusters; i++){
//			lambda.at(i) = gamma_temp.at<double>(i,0)/sum_gamma_temp.at(0);
//			cout<<"new_lambda: "<<lambda.at(i)<<endl;
//		}
		// ****************** new mean ************************
		cv::Mat gamma_trans;
		cv::transpose(gamma, gamma_trans);
		cv::Mat new_mean (n_clusters, dataset.at(0).size(), CV_64FC1);
		cv::Mat new_mean_temp (dataset.size(), dataset.at(0).size(), CV_64FC1);
		for (int i=0; i<n_clusters; i++){
			for (int j=0; j<dataset.size(); j++){
				for (int k=0; k<dataset.at(0).size(); k++){
					new_mean_temp.at<double>(j,k) = gamma_trans.at<double>(i,j)*dataset[j][k];
				}
			}
			for(int j=0; j<dataset.at(0).size(); j++){
				vector<double> c;
				cv::reduce(new_mean_temp(Range::all(), Range(j,j+1)),c,0,CV_REDUCE_SUM);
				new_mean.at<double>(i,j) = c.at(0);
			}
		}
		// ******************* new covar ************************
		vector<Mat> ch;
		for (int i=0; i<n_clusters; i++){
			cv::Mat temp1(dataset.size(), dataset.at(0).size(), CV_64FC1);
			cv::Mat temp2(dataset.size(), dataset.at(0).size(), CV_64FC1);
			cv::Mat temp1_1(dataset.at(0).size(),1, CV_64FC1);
			cv::Mat temp2_2(dataset.at(0).size(),1, CV_64FC1);
			for (int j=0; j<dataset.size(); j++){
				Mat temp=Mat(1,dataset.at(0).size(),CV_64FC1);
				for (int l = 0; l < dataset.at(j).size(); l++)
				{
					temp.at<double>(0,l)= dataset[j][l];
				}
				temp1.row(j) = temp.row(0)-new_mean.row(i);
				temp2.row(j) = temp1.row(j)*gamma.at<double>(j,i);
			}
			for (int k=0; k<dataset.at(0).size(); k++){
				vector<double> d,e;
				cv::reduce(temp1(Range::all(), Range(k,k+1)),d,0,CV_REDUCE_SUM);
				cv::reduce(temp2(Range::all(), Range(k,k+1)),e,0,CV_REDUCE_SUM);
				temp1_1.row(k) = d.at(0);
				temp2_2.row(k) = e.at(0);
			}
			cv::Mat temp2_2_trans = temp2_2.t();
			cv::Mat new_cov_temp = (temp1_1*temp2_2_trans)/sum_gamma_temp.at(0);
			ch.push_back(new_cov_temp);
		}
		cv::Mat new_covar;
		merge(ch,new_covar);
		// ******************* updates ************************
		cv::Mat updated_mean (n_clusters, 100, CV_64FC1);
		for (int i=0; i<new_mean.rows; i++){
			double min_mean, max_mean;
			cv::minMaxLoc( new_mean.row(i), &min_mean, &max_mean);
			for (int j=0; j<new_mean.cols; j++){
				updated_mean.at<double>(i,j) = 255*((new_mean.at<double>(i,j) - double(min_mean))/(double(max_mean)-double(min_mean)));
			}
		}
		vector<Mat> covar_cluster;
		cv::Mat Mat_cluster[n_clusters];
		split(new_covar, Mat_cluster);
		for (int k=0; k<n_clusters; k++){
			double covar_min, covar_max;
			cv::minMaxLoc( Mat_cluster[k], &covar_min, &covar_max);  // check function !!!!!
			for (int i=0; i<Mat_cluster[k].rows; i++){
				for (int j=0; j<Mat_cluster[k].cols; j++){
					if (i!=j){
						Mat_cluster[k].at<double>(i,j) = 0;
					}
					else{
						Mat_cluster[k].at<double>(i,j) =10000 + ((Mat_cluster[k].at<double>(i,j) - double(covar_min))/(double(covar_max)-double(covar_min)));
						}
					}
				}
			double covar_min_af, covar_max_af;
			cv::minMaxLoc( Mat_cluster[k], &covar_min_af, &covar_max_af);  // check function !!!!!
			covar_cluster.push_back(Mat_cluster[k]);
		}
		cv::Mat updated_covar;
		merge(covar_cluster,updated_covar);
		return make_pair(updated_mean.clone(), updated_covar.clone());
	}

	Mat calNorm(vector<vector<int> >test, vector<double> lambda, Mat mean, Mat covar)
	{
		Mat pdf(test.size(), n_clusters, CV_64FC1);
		Mat sum_pdf(test.size(), 1, CV_64FC1);

		Mat covar_cluster[n_clusters];
		split(covar, covar_cluster);

		for (int i=0; i<test.size(); i++){
			for (int j=0; j<n_clusters; j++){
				double arr[test[i].size()];
				std::copy(test[i].begin(), test[i].end(), arr);
				cv::Mat single_dataset(1, 100, CV_64FC1, arr);
				single_dataset = single_dataset.reshape(1,1);
				cv::Mat d = (single_dataset - mean.row(j))*(covar_cluster[j].inv());
				cv::Mat d1 =  d*(single_dataset - mean.row(j)).t();

				pdf.at<double>(i,j)= double(lambda[j])*double(exp(d1.at<double>(0,0)*(-0.5)));
			}
			vector<double> a;
			cv::reduce(pdf(Range(i,i+1), Range::all()), a,1,CV_REDUCE_SUM);
			sum_pdf.at<double>(i,0) = a.at(0);
		}
		return sum_pdf.clone();
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

	void save_mean_images(Mat all_means, int class_name){
		for (int i=0; i<n_clusters; i++){
			Mat mean_face_cl = all_means.row(i);
			mean_face_cl.convertTo(mean_face_cl, CV_8UC1);
			mean_face_cl=mean_face_cl.reshape(1,10);
			string name;
			if (class_name==0){
				name = "/home/arshita/workspace/FaceClassification/mean_face_cl_";
			}
			else
				name = "/home/arshita/workspace/FaceClassification/mean_nonface_cl_";
			stringstream ss;
			ss<<i;
			name = name+ ss.str();
			name+=".png";
			imwrite(name, mean_face_cl);
		}
	}

	void train(vector<vector<int> > train_face,
			vector<vector<int> >train_nonface,
			vector<vector<int> >test_face,
			vector<vector<int> >test_nonface){

		clock_t ti;
		vector<double> lambda (n_clusters, double(1)/double(n_clusters));
		vector<double> lambda_nonface (n_clusters, double(1)/double(n_clusters));

		pair<Mat, Mat> init_face = initialize_params();
		Mat mean_face = init_face.first;
		Mat covar_face = init_face.second;
		pair<Mat, Mat> init_nonface = initialize_params();
		Mat mean_nonface = init_nonface.first;
		Mat covar_nonface = init_nonface.second;

		std::cout<<"Parameters initialized"<<endl;
		int iterations = 5;
		for (int i=0; i<iterations; i++){
			std::cout<<"\niteration no: "<<i<<endl;

			// ******************* face *************************
			ti = clock();
			pair<Mat, Mat> gamma_all = e_step(train_face, lambda, mean_face, covar_face);
			cv::Mat gamma = gamma_all.first;
			cv::Mat gamma_temp = gamma_all.second;
			ti = clock()-ti;
			std::cout<<"E-step complete for Face "<<"in time: "<<((float)ti)/CLOCKS_PER_SEC<<" seconds"<<endl;
			vector<double> sum_gamma_temp;
			cv::reduce(gamma_temp.col(0), sum_gamma_temp,0,CV_REDUCE_SUM);
			for (int i =0; i<n_clusters; i++){
				lambda.at(i) = gamma_temp.at<double>(i,0)/sum_gamma_temp.at(0);
			}
			ti = clock();
			pair<Mat, Mat> new_= m_step(train_face, gamma, gamma_temp);
			mean_face = new_.first;
			covar_face = new_.second;
			ti = clock() - ti;
			std::cout<<"M-step complete for Face "<<"in Time: "<<((float)ti)/CLOCKS_PER_SEC<<" seconds"<<endl;
			// ****************** non face **********************
			pair<Mat, Mat> gamma_all_non = e_step(train_nonface, lambda_nonface, mean_nonface, covar_nonface);
			cv::Mat gamma_non = gamma_all_non.first;
			cv::Mat gamma_non_temp = gamma_all_non.second;
			std::cout<<"E-step complete for Non-Face"<<endl;

			vector<double> sum_gamma_non_temp;
			cv::reduce(gamma_non_temp.col(0), sum_gamma_non_temp,0,CV_REDUCE_SUM);
			for (int i =0; i<n_clusters; i++){
				lambda_nonface.at(i) = gamma_non_temp.at<double>(i,0)/sum_gamma_non_temp.at(0);
			}

			pair<Mat, Mat> new_non_= m_step(train_nonface, gamma_non, gamma_non_temp);
			mean_nonface = new_non_.first;
			covar_nonface = new_non_.second;
			std::cout<<"M-step complete for Non-Face"<<endl;
		}
		std::cout<<"\nStart Writing Mean Face Images"<<endl;
		save_mean_images(mean_face, 0);
		std::cout<<"\nStart Writing Mean Non-Face Images"<<endl;
		save_mean_images(mean_nonface, 1);
		std::cout<<"\nAll 'mean' Images written"<<endl;

		cv::Mat logpdf_face_wrt_face = calNorm(test_face, lambda, mean_face, covar_face);
		cv::Mat logpdf_face_wrt_nonface = calNorm(test_face, lambda, mean_nonface, covar_nonface);
		cv::Mat logpdf_nonface_wrt_face = calNorm(test_nonface, lambda, mean_face, covar_face);
		cv::Mat logpdf_nonface_wrt_nonface = calNorm(test_nonface, lambda, mean_nonface, covar_nonface);

		vector<float> pos_testface_face, pos_testface_nonface, pos_testnonface_face, pos_testnonface_nonface;
		pair<vector<float>,vector<float> > pos_testface = posterior(logpdf_face_wrt_face, logpdf_face_wrt_nonface);
		pos_testface_face = pos_testface.first;
		pos_testface_nonface = pos_testface.second;
		pair<vector<float>,vector<float> > pos_testnonface = posterior(logpdf_nonface_wrt_face, logpdf_nonface_wrt_nonface);
		pos_testnonface_face = pos_testnonface.first;
		pos_testnonface_nonface = pos_testnonface.second;

		int count = 0;
		for (int i =0; i<pos_testface_face.size(); i++){
			if (pos_testface_face.at(i)>pos_testface_nonface.at(i))
				count+=1;
		}
		std::cout<<"\nTrue Positive Rate: "<< float(count)/(pos_testface_face.size())<<endl;

		int count_neg = 0;
		for (int i =0; i<pos_testface_face.size(); i++){
			if (pos_testnonface_nonface.at(i)>pos_testnonface_face.at(i))
				count_neg+=1;
		}
		std::cout<<"\nTrue Negative Rate: "<< float(count_neg)/(pos_testface_face.size())<<endl;
		}
};
