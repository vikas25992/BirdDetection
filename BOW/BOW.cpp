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

void loadImagesFromDirectory(const string& filepath,vector<Mat>& imageFiles);
const Mat K_means(const Mat descriptors, Mat vocabulary);
Mat BOWDescriptors();
Size imagesize(100,100);

void loadImagesFromDirectory(const string& filepath,vector<Mat>& imageFiles)
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
		imageFiles.push_back(image);
	   }

	}
}

//Read from a file
static bool readVocabulary( const string& filename, Mat& vocabulary )
{
    FileStorage fs( filename, FileStorage::READ );
    if( fs.isOpened() )
    {
	fs["vocabulary"] >> vocabulary;
       // cout << "done" << endl;
	return true;
    }
    return false;
}

const Mat K_means(Mat descriptors, Mat vocabulary)	
{
	int vocabulary_size = 500;
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

Mat BOWDescriptors(Mat Vocabulary,vector<Mat> train_images )
{
	Mat img, train_hist;
	// Building Histograms
	cout << "===========================\n";

	std::vector< DMatch > matches;
	// Matching centroids with training images
	//std::vector<DMatch> trainin_set_matches;

	//Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
	SiftFeatureDetector featureDetector;
	Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
	Ptr<BOWImgDescriptorExtractor> bowExtractor;

	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

	bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

	bowExtractor->setVocabulary(Vocabulary);

	cout << "creating Histograms "<<endl;

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
		featureDetector.detect( train_images.at(i), keypoints );
		bowExtractor->compute( train_images.at(i), keypoints, descriptors);
		train_hist.push_back(descriptors);
		cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
		}		
	}
	
	return train_hist;
}

int main(int argc,char **argv)
{
	Mat Descriptors,Vocabulary, Mauersegler,Birds,BackGround,Mauersegler_hist,Bird_hist,BG_hist;
	vector<Mat> Mauersegler_images,Bird_images,BG_images;

	const string Mauersegler_address = "/home/vikas/Desktop/Main/Dataset/Train/Mauersegler";
	const string Bird_address = "/home/vikas/Desktop/Main/Dataset/Train/Birds";
	const string Background_address = "/home/vikas/Desktop/Main/Dataset/Train/Back_ground";

	loadImagesFromDirectory(Mauersegler_address,Mauersegler_images);
	loadImagesFromDirectory(Bird_address,Bird_images);
	loadImagesFromDirectory(Background_address,BG_images);

	//Reading Sift desc of Mauersegler
	FileStorage fs1("/home/vikas/Desktop/Main/trail3/Mauersegler.dat",FileStorage::READ);
	fs1["feature_descriptors"] >> Mauersegler;
	cout<<"\n"<<Mauersegler.rows;
	Descriptors.push_back(Mauersegler);

	//Reading Sift desc of Birds
	FileStorage fs2("/home/vikas/Desktop/Main/trail3/Birds.dat",FileStorage::READ);
	fs2["feature_descriptors"] >> Birds;
	cout<<"\n"<<Birds.rows;
	Descriptors.push_back(Birds);

	//Reading Sift desc of BackGround
	FileStorage fs3("/home/vikas/Desktop/Main/trail3/BackGround.dat",FileStorage::READ);
	fs3["feature_descriptors"] >> BackGround;
	cout<<"\n"<<BackGround.rows;
	Descriptors.push_back(BackGround);

	if(readVocabulary("/home/vikas/Desktop/Main/trail3/vocabulary.dat", Vocabulary))
	{
		cout << "Visual Vocabulary read from file successfully!\n";
	}
	else
	{
		Vocabulary = K_means(Descriptors,Vocabulary);

		FileStorage fs("/home/vikas/Desktop/Main/trail3/vocabulary.dat",FileStorage::WRITE);
	
		fs<<"vocabulary"<<Vocabulary;
	
		fs.release();
	}
	
	if(readVocabulary("/home/vikas/Desktop/Main/trail3/Mauersegler_hist.dat", Mauersegler_hist))
	{
		cout << "Mauersegler_hist read from file successfully!\n";
	}
	else
	{
		Mauersegler_hist = BOWDescriptors(Vocabulary,Mauersegler_images );
		
		FileStorage fs4("/home/vikas/Desktop/Main/trail3/Mauersegler_hist.dat",FileStorage::WRITE);
	
		fs4<<"Mauersegler_hist"<<Mauersegler_hist;
	
		fs4.release();
	}
	if(readVocabulary("/home/vikas/Desktop/Main/trail3/Birds_hist.dat", Bird_hist))
	{
		cout << "Bird_hist read from file successfully!\n";
	}
	else
	{
		Bird_hist = BOWDescriptors(Vocabulary,Bird_images );
		
		FileStorage fs5("/home/vikas/Desktop/Main/trail3/Bird_hist.dat",FileStorage::WRITE);
	
		fs5<<"Bird_hist"<<Bird_hist;
	
		fs5.release();
	}
	if(readVocabulary("/home/vikas/Desktop/Main/trail3/BG_hist.dat", BG_hist))
	{
		cout << "BG_hist read from file successfully!\n";
	}
	else
	{
		BG_hist = BOWDescriptors(Vocabulary,BG_images );
		
		FileStorage fs6("/home/vikas/Desktop/Main/trail3/BG_hist.dat",FileStorage::WRITE);
	
		fs6<<"BG_hist"<<BG_hist;
	
		fs6.release();
	}
	
}
