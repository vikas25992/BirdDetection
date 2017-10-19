#ifndef DWORD
#define WINAPI
typedef unsigned long DWORD;
#endif

#include<iostream>
#include<fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>

#include <sstream>
#include<string>

using namespace std;
using namespace cv;

void loadImagesFromDirectory(const string& filepath,vector<Mat>& imageFiles);
static bool writeVocabulary( const string& filename, const Mat& vocabulary );
static bool readVocabulary( const string& filename, Mat& vocabulary );
void Sift_Extractor(vector<Mat>& train_images, Mat& feature_descriptors);
const Mat K_means(const Mat descriptors, Mat vocabulary);
Mat Histograms(Mat& Vocabulary, Mat& hist, vector<Mat>& images);
Size imagesize(100,100);


//Read images from the file
	void loadImagesFromDirectory(const string& filepath,vector<Mat>& imageFiles, vector<Mat>& train)
	{
		Mat image;
		vector<String> filesInFolder;
		// Reading set of images from a folder
		glob(filepath, filesInFolder, false);
		for (size_t i = 0; i < filesInFolder.size(); i++)
		{
			image = imread(filesInFolder[i], 0);
			if(image.empty())
		    {
			cout<<"\nCan't read the images\n"<<filepath;
			continue;

		    }

		else
		   {	 
			resize(image,image,imagesize);
			//cout<<""<<filesinFolder[i]<<endl;
			imageFiles.push_back(image);
			train.push_back(image);
		   }

			//train_set.push_back(image);
		}
	}


//Write to a file
	static bool writeVocabulary( const string& filename, const Mat& vocabulary )
	{
	    FileStorage fs( filename, FileStorage::WRITE );
	    if( fs.isOpened() )
	    {
		fs << "vocabulary" << vocabulary;
		return true;
	    }
	    return false;
	}

//Read from a file
	static bool readVocabulary( const string& filename, Mat& vocabulary )
	{
	    FileStorage fs( filename, FileStorage::READ );
	    if( fs.isOpened() )
	    {
		fs["vocabulary"] >> vocabulary;
		return true;
	    }
	    return false;
	}

//SIFT feature extractor
	void Sift_Extractor(vector<Mat>& train_images, Mat& feature_descriptors)	
	{
		SiftFeatureDetector detector;
		SiftDescriptorExtractor extractor;

		Mat descriptors;

		for(int i=0; i<train_images.size();i++)
		{	
			Mat image = train_image.at(i);
			cvtColor(image,image,CV_BGR2GRAY);
			vector<cv::KeyPoint> keypoints;
			detector.detect(image,keypoints);
			extractor.compute(image, keypoints, descriptors);
			feature_descriptors.push_back(descriptors);
			cout << "Extracting image #"<<i << "/" << train_images.size() << "\r" << std::flush;
		}

	    	cout <<endl << feature_descriptors.size() << " features extracted for training images.\n";
	}

//K-means clustering
	const Mat K_means(const Mat descriptors, Mat vocabulary)	
	{
		int vocabulary_size = 250;
		TermCriteria terminate_criterion;
		terminate_criterion.epsilon = FLT_EPSILON; //FLT_EPSILON = 1.19209e-07
		BOWKMeansTrainer bowTrainer( vocabulary_size, terminate_criterion, 3, KMEANS_PP_CENTERS );
		for (int i=0;i<descriptors.size().height;i++)
		{
			Mat current_descriptor = descriptors.row(i);
			bowTrainer.add(current_descriptor);
			cout << "Adding Feature #" << i << " to Bag-Of-Words K-Means Trainer ...  \r" << std::flush;
		}
		cout << "\nClustering... Please Wait ...\n";
		vocabulary = bowTrainer.cluster();
		cout << "\nSIFT Features Clustered in " << vocabulary.size() << " clusters." << endl;
		return vocabulary;
	
	}

//creating Histograms

/*	Mat Histograms(Mat& Vocabulary, Mat& hist, vector<Mat>& images)
	{
		// Building Histograms
		cout << "===========================\n";

		std::vector< DMatch > matches;
		// Matching centroids with training images
		std::vector<DMatch> trainin_set_matches;

		Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
		Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
		Ptr<BOWImgDescriptorExtractor> bowExtractor;

		Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

		bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

		bowExtractor->setVocabulary(Vocabu lary);
	
		cout << "creating Histograms "<<endl;
		
		Mat img;

		for (int i=0;i<images.size();i++)
		{
			img = images.at(i);
			vector<cv::KeyPoint> keypoints;
			// Each descriptor is histogram for the image
			Mat descriptors;
			if(img.empty())
			{
			cout<<"error"<<endl;	
			}
			else
			{
			featureDetector->detect( images.at(i), keypoints );
			bowExtractor->compute( images.at(i), keypoints, descriptors);
			hist.push_back(descriptors);
			cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
			}		
		}
		return hist;
	}

*/
//confusion matrix

struct category
{
	int count;
	string name;
	int ind;
};

struct confusion
{
	int ind;
	string name;
	vector<category> categories;
};

using namespace cv;
using namespace std;

int main(int argc,char **argv)
{
	vector<Mat> Mauersegler_images, Bird_images, bg_images, train_images,test_images;
	vector<int> train_labels;
	Mat image, feature_descriptors, vocabulary, histograms, train_hist, test_hist;

	const string Mauersegler_address = "/home/vikas/Desktop/Main/Dataset/Train/Mauersegler";
	const string Bird_address = "/home/vikas/Desktop/Main/Dataset/Train/Birds";
	const string Background_address = "/home/vikas/Desktop/Main/Dataset/Train/Back_ground";
	const string Test_address = "/home/vikas/Desktop/Main/Dataset/Test";

	cout << "Reading Mauersegler data"<<endl;
	loadImagesFromDirectory(Mauersegler_address,Mauersegler_images,train_images);
	train_labels.assign(Mauersegler_images.size(), 1);

	cout << "Reading Birds images"<<endl;
	loadImagesFromDirectory(Bird_address,Bird_images,train_images);
	train_labels.assign(Bird_images.size(), 0);

	cout << "Reading Background images"<<endl;
	loadImagesFromDirectory(Background_address,bg_images,train_images);
	train_labels.assign(bg_images.size(), -1);
	
	cout << "Extracting SIFT Features of training images ...\n";


	if(!readVocabulary("vocabulary.dat", vocabulary))
	{
	//SIFT features extractor

		Sift_Extractor(Mauersegler_images, feature_descriptors);
		Sift_Extractor(Bird_images, feature_descriptors);
		Sift_Extractor(bg_images, feature_descriptors);	

	//using KNN

		vocabulary = K_means(feature_descriptors,vocabulary);

	    	if( !writeVocabulary("vocabulary.dat", vocabulary) )
	    	{
	    		cout << "Error: file " << "vocabulary.dat" << " can not be opened to write" << endl;
	    		exit(-1);
	    	}
	}
	else
	    cout << "Visual Vocabulary read from file successfully!\n";


	// Building Histograms
	cout << "===========================\n";

	std::vector< DMatch > matches;
	// Matching centroids with training images
	std::vector<DMatch> trainin_set_matches;

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
	Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
	Ptr<BOWImgDescriptorExtractor> bowExtractor;

	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

	bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

	bowExtractor->setVocabulary(vocabulary);

	cout << "creating Histograms "<<endl;

	if(!readVocabulary( "train_hist.dat", train_hist) )
	{
	
	Mat img;

	for (int i=0;i<train_images.size();i++)
	{
		img = train_images.at(i);
		vector<cv::KeyPoint> keypoints;
		// Each descriptor is histogram for the image
		Mat descriptors;
		if(img.empty())
		{
		cout<<"error"<<endl;	
		}
		else
		{
		featureDetector->detect( train_images.at(i), keypoints );
		bowExtractor->compute( train_images.at(i), keypoints, descriptors);
		train_hist.push_back(descriptors);
		cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
		}		
	}
	writeVocabulary("train_hist.dat", train_hist);
	}
	else
	{
	cout << "Train Histograms read from file successfully!\n";
	}
	
/*	if(!readVocabulary( "train_hist.dat", train_hist) )
	{
		histograms = Histograms(vocabulary,histograms,Mauersegler_images);
		train_hist.push_back(histograms);
		histograms = Histograms(vocabulary,histograms,Bird_images);
		train_hist.push_back(histograms);
		histograms = Histograms(vocabulary,histograms,bg_images);
		train_hist.push_back(histograms);

	writeVocabulary("train_hist.dat", train_hist);
	}else{
	cout << "Train Histograms read from file successfully!\n";
	}
	
	cout << "Reading Test data"<<endl;
	
	vector<String> filesInFolder;
	
	glob(Test_address, filesInFolder, false);

	for (size_t i = 0; i < filesInFolder.size(); i++)
	{
		image = imread(filesInFolder[i], 0);

		if(image.empty())
		    {
			cout<<"\nCan't read the images\n";

		    }

		else
		   {	 
			resize(image,image,imagesize);
			//cout<<""<<filesinFolder[i]<<endl;
			test_images.push_back(image);
		   }
	}

	cout << "\nBuilding Histograms for test set :\n"; 

	if( !readVocabulary( "test_hist.dat", test_hist) )
	{	    
		Mat img;

		for (int i=0;i<test_images.size();i++)
		{
			img = test_images.at(i);
			vector<cv::KeyPoint> keypoint;
			// Each descriptor is histogram for the image
			Mat test_descriptors;
			if(img.empty())
			{
			cout<<"error"<<endl;	
			}
			else
			{
			featureDetector->detect( test_images.at(i), keypoint );
			bowExtractor->compute( test_images.at(i), keypoint, test_descriptors);
			test_hist.push_back(test_descriptors);
			cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << test_descriptors.size() << "\r" << std::flush;
			}		
		}
		writeVocabulary("../test_hist.dat", test_hist);
	}
	else
	{
		cout << "Test Histograms read from file successfully!\n";
	}

	// Set up SVM's parameters

*/	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-7);
	params.C = 120;

	std::vector<Ptr<CvSVM> > svms;
	CvSVM *main_svm;
	main_svm = new CvSVM;
	
	main_svm->train(train_hist,train_labels,Mat(),Mat(),params);

	main_svm->save("trail.xml");	
	
	cout << "SVM training done!\n";

	
	string name[3] = {"Mauersegler","Bird","Background"};
	vector<confusion> all;
	for(int i=0;i<3; i++)
	{
		confusion current_conf;
		current_conf.ind = i;
		current_conf.name = name[i];
		
		vector<category> cat;
		for(int j=0; j<3; j++)
		{	
			category current_cat;
			current_cat.ind = j;
			current_cat.name = name[j];
			current_cat.count = 0;
			cat.push_back(current_cat);
		}
		current_conf.categories = cat;
		all.push_back(current_conf);
	}

	//test SVM

/*	int misclassified_count = 0;

	for (int i=0;i<test_images.size();i++)
	{
		int best_cat = -1;
		float best_response=-1000000000;
		for (int ind=0;ind<3;ind++)
		{
	    		float response = (svms.at(ind)->predict(test_hist.row(i) , true));
	    		if (response > best_response)
			{
	    			best_cat  = ind;
	    			best_response = response;
    			}
    		}
	}*/																																																																																								

}



