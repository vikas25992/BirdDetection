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

void loadImagesFromDirectory(const string& filepath);
void Sift_Extractor(vector<Mat>& train_images, Mat& feature_descriptors);
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
	    }

	else
	   {	 
		resize(image,image,imagesize);
		imageFiles.push_back(image);
	   }

	}
}

//SIFT feature extractor
void Sift_Extractor(vector<Mat>& train_images, Mat& feature_descriptors)	
{
	SiftFeatureDetector detector;
	SiftDescriptorExtractor extractor;

	Mat descriptors;

	for(int i=0; i<train_images.size();i++)
	{
		vector<cv::KeyPoint> keypoints;
		detector.detect(train_images.at(i),keypoints);
		extractor.compute(train_images.at(i), keypoints, descriptors);
		feature_descriptors.push_back(descriptors);
		cout << "Extracting image #"<<i << "/" << train_images.size() << "\r" << std::flush;
	}

	FileStorage fs("/home/vikas/Desktop/Main/trail3/BackGround.dat",FileStorage::WRITE);

    	fs<<"feature_descriptors"<<feature_descriptors;
    	
    	fs.release();

    	cout <<endl << feature_descriptors.size() << " features extracted for training images.\n";
}

int main(int argc,char **argv)
{
	vector<Mat> bg_images;
	Mat feature_descriptors;
	const string Background_address = "/home/vikas/Desktop/Main/Dataset/Train/Back_ground";

	cout << "Reading Background images"<<endl;
	loadImagesFromDirectory(Background_address,bg_images);

	Sift_Extractor(bg_images,feature_descriptors);
	
	return 1;
}

