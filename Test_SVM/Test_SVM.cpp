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
#include <string>

using namespace std;
using namespace cv;

static bool readVocabulary( const string& filename, Mat& vocabulary );
void loadImagesFromDirectory(const string& filepath,vector<Mat>& test_images);
Size imagesize(100,100);

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
	
float Test(Mat test_image, Mat vocabulary)
{
	float label;

	CvSVM svm;
	svm.load("/home/vikas/Desktop/Main/trail3/classifier.xml","circuit");
	
	std::vector< DMatch > matches;
	//Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
	SiftFeatureDetector detector;
	Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
	Ptr<BOWImgDescriptorExtractor> bowExtractor;

	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

	bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

	bowExtractor->setVocabulary(vocabulary);


	vector<cv::KeyPoint> keypoint;
	// Each descriptor is histogram for the image
	Mat test_descriptors;
	if(test_image.empty())
	{
		cout<<"error"<<endl;	
	}
	else
	{
	//featureDetector->detect( test_images.at(i), keypoint );
		detector.detect(test_image, keypoint );	//error of the program
		bowExtractor->compute( test_image, keypoint, test_descriptors);
		label = svm.predict(test_descriptors, false);
		//cout<<"\n value:"<<predLabel;
	//test_hist.push_back(test_descriptors);
	}		
	
	return label;
}

int main(int argc,char **argv)
{
	vector<Mat> MauerseglerTestImages,BirdTestImages,BGTestImages;
	float PredictedLabel;
	Mat image, test_hist,Vocabulary;
	unsigned int categories = 3;
	int ConfusionMatrix[3][3]={0,0,0,0,0,0,0,0,0};

	const string Mauersegler_address = "/home/vikas/Desktop/Main/Dataset/Test/Mauersegler";
	const string Bird_address = "/home/vikas/Desktop/Main/Dataset/Test/Birds";
	const string BG_address = "/home/vikas/Desktop/Main/Dataset/Test/BG";

	loadImagesFromDirectory(Mauersegler_address,MauerseglerTestImages);
	loadImagesFromDirectory(Bird_address,BirdTestImages);
	loadImagesFromDirectory(BG_address,BGTestImages);

	if(!readVocabulary("/home/vikas/Desktop/Main/trail3/vocabulary.dat", Vocabulary))
	{	
		cout<<"Vocabulary not found"<<endl;
	}
	else
	{
		cout<<"Reading Vocabulary"<<endl;
	}

	for(int i=0; i<MauerseglerTestImages.size();i++)
	{
		PredictedLabel = Test(MauerseglerTestImages.at(i),Vocabulary);
		if(PredictedLabel == 1)
		{ConfusionMatrix[0][0]++;}
		else if(PredictedLabel == 2)
		{ConfusionMatrix[0][1]++;}
		else if(PredictedLabel == 3)
		{ConfusionMatrix[0][2]++;}
	}
	for(int i=0; i<BirdTestImages.size();i++)
	{
		PredictedLabel = Test(BirdTestImages.at(i),Vocabulary);
		if(PredictedLabel == 1)
		{ConfusionMatrix[1][0]++;}
		else if(PredictedLabel == 2)
		{ConfusionMatrix[1][1]++;}
		else if(PredictedLabel == 3)
		{ConfusionMatrix[1][2]++;}
	}
	for(int i=0; i<BGTestImages.size();i++)
	{
		PredictedLabel = Test(BGTestImages.at(i),Vocabulary);
		if(PredictedLabel == 1)
		{ConfusionMatrix[2][0]++;}
		else if(PredictedLabel == 2)
		{ConfusionMatrix[2][1]++;}
		else if(PredictedLabel == 3)
		{ConfusionMatrix[2][2]++;}
	}

	cout<<"Confusion Matrix"<<endl;
	cout<<"\tMauersegler\t\tBirds\t\tOther"<<endl;
	
	for(unsigned int i=0; i<categories; i++)
	{
		for(unsigned int j=0; j<categories; j++)
		{
			cout<<"\t\t"<<ConfusionMatrix[i][j];
		}	
		cout<<"\n";
	}

	//cout<<"\t"<<ConfusionMatrix[0][0]<<endl;

	return 0;

}
