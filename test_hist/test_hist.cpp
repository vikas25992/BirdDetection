#include<iostream>
#include<fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <vector>

#include <sstream>
#include<string>

using namespace std;
using namespace cv;

void loadImagesFromDirectory(const string& filepath,vector<Mat>& test_images);
static bool writeVocabulary( const string& filename, const Mat& vocabulary );
static bool readVocabulary( const string& filename, Mat& vocabulary );
Size imagesize(100,100);

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
       // cout << "done" << endl;
	return true;
    }
    return false;
}

//Read images from the file
void loadImagesFromDirectory(const string& filepath,vector<Mat>& test_images)
{
	Mat image;
	vector<String> filesInFolder;
	// Reading set of images from a folder
	glob(filepath, filesInFolder, false);
	for (size_t i = 0; i < filesInFolder.size(); i++)
	{
		image = imread(filesInFolder[i], 1);
		if(image.empty())
	    {
		cout<<"\nCan't read the images\n"<<filepath;

	    }

	else
	   {	 
		resize(image,image,imagesize);
		cvtColor(image,image,CV_BGR2GRAY);
		//cout<<""<<filesInFolder[i]<<"\n";
		test_images.push_back(image);
	   }

	}
}


int main(int argc,char **argv)
{
	vector<Mat> test_images;
	Mat image, vocabulary, test_hist;
	const string Test_address = "/home/vikas/Desktop/Main/Dataset/Test";
	
	if(!readVocabulary("/home/vikas/Desktop/Main/trail3/vocabulary.dat", vocabulary))
	{	
		cout<<"Vocabulary not found"<<endl;
	}
	else
	{
		cout<<"Reading Vocabulary"<<endl;
	}
	cout << "Reading Test data"<<endl;
	loadImagesFromDirectory(Test_address,test_images);

		// Building Histograms
	cout << "===========================\n";
	
	std::vector< DMatch > matches;
	//Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
	SiftFeatureDetector detector;
	Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
	Ptr<BOWImgDescriptorExtractor> bowExtractor;

	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

	bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

	bowExtractor->setVocabulary(vocabulary);

	if(!readVocabulary( "/home/vikas/Desktop/Main/trail3/test_hist.dat", test_hist) )
	{	
	  for (int i=0;i<test_images.size();i++)
	  {
		//image = test_images.at(i);
		vector<cv::KeyPoint> keypoint;
		// Each descriptor is histogram for the image
		Mat test_descriptors;
		if(test_images.at(i).empty())
		{
		cout<<"error"<<endl;	
		}
		else
		{
		//featureDetector->detect( test_images.at(i), keypoint );
		detector.detect(test_images.at(i), keypoint );	//error of the program
		bowExtractor->compute( test_images.at(i), keypoint, test_descriptors);
		test_hist.push_back(test_descriptors);
		cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << test_descriptors.size() << "\r" << std::flush;
		}		
	   }
	writeVocabulary("/home/vikas/Desktop/Main/trail3/test_hist.dat", test_hist);
	}
	else 
	{
		cout << "Test Histograms read from file successfully!\n";
	}

		//cout<<"good"<<endl;

	FileStorage fs1("/home/vikas/Desktop/Main/trail3/test_hist.dat",FileStorage::WRITE);

    	fs1<<"test_hist"<<test_hist;
    	
    	fs1.release();	
	return 1;
}

