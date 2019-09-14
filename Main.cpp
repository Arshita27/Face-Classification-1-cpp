#include <time.h>
#include <iostream>

#include "SimpleGaussianModel.cpp"
#include "MixtureGaussianModel.cpp"
#include "utils.h"

using namespace cv;
using namespace std;


int main(){
	// ROOT DIR: set root directory where Dataset is saved.
	string ROOT_DIR = "/home/arshita/Desktop/Dataset/";
	// SAVE_DIR: set directory where mean (calculated) images should be saved.
	string SAVE_DIR = "/home/arshita/workspace/FaceClassification/";
	// Model: set Model name that needs to be implemented.
	string Model = "Mixture of Gaussian";

	vector<vector<int> > train_face, train_nonface, test_face, test_nonface;
	vector<string> FULL_PATH = get_image_Path(ROOT_DIR);

	train_face = get_images(FULL_PATH[0]);
	train_nonface = get_images(FULL_PATH[1]);
	test_face = get_images(FULL_PATH[2]);
	test_nonface = get_images(FULL_PATH[3]);

	// Simple Gaussian
	if (Model == "Simple Gaussian"){
		SimpleGaussian sim_gaus(SAVE_DIR);
		sim_gaus.train(train_face, train_nonface, test_face, test_nonface);
	}

	// Mixture of Gaussian
	if (Model == "Mixture of Gaussian"){
		int num_clusters = 3;
		int total_iterations = 5;
		MixtureGaussian mix_gaus(SAVE_DIR, num_clusters, total_iterations);
		mix_gaus.train(train_face, train_nonface, test_face, test_nonface);
	}
}
