#include<iostream>
#include<fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <vector>

#include <sstream>
#include<string>

using namespace std;
using namespace cv;

int main()
{

	Mat train_hist,labels,Mauersegler,Birds,BG;


	FileStorage fs1("Mauersegler_hist.dat",FileStorage::READ);
	fs1["Mauersegler_hist"]>>Mauersegler;
	train_hist.push_back(Mauersegler);
	Mat Mauersegler_labels=Mat::ones(Mauersegler.rows,1,CV_32SC1);
	labels.push_back(Mauersegler_labels);	
	cout<<"training data:"<<train_hist.rows<<","<<train_hist.cols<<"\n Label:"<<labels.rows;
	fs1.release();

	FileStorage fs2("Bird_hist.dat",FileStorage::READ);
	fs2["Bird_hist"]>>Birds;
	train_hist.push_back(Birds);
	Mat Birds_labels = Mat::ones(Birds.rows,1,CV_32SC1);
	Birds_labels.setTo((float)2);
	labels.push_back(Birds_labels);
	cout<<"training data:"<<train_hist.rows<<","<<train_hist.cols<<"\n Label:"<<labels.rows;
	fs2.release();

	FileStorage fs3("BG_hist.dat",FileStorage::READ);
	fs3["BG_hist"]>>BG;
	train_hist.push_back(BG);
	Mat BG_labels = Mat::ones(BG.rows,1,CV_32SC1);
	BG_labels.setTo((float)3);
	labels.push_back(BG_labels);
	cout<<"training data:"<<train_hist.rows<<","<<train_hist.cols<<"\n Label:"<<labels.rows;
	fs3.release();

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit =  cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	
	//Train array:DD array with responsehistograms values
	
	Mat svmData;
	train_hist.convertTo(svmData,CV_32F);
	Mat labelData;
	labelData=labels.t(); //labels: Array of labels
	
	

	//SVM training:
	
	CvSVM svm;
	params=svm.get_params();
	
	svm.train_auto(svmData, labelData, Mat(), Mat(), params, 10);
	svm.save("classifier.xml","circuit");
	
	//End of training

	
	return 0;
	

}
