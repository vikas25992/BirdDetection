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

void loadImagesFromDirectory(const string& filepath,vector<Mat>& test_images);
Size imagesize(100,100);

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
	vector<Mat> mauerseglerTestImages;
	Mat TestLabels, vocabulary;
	const string Test_address = "/home/vikas/Desktop/Main/Dataset/Test/Mauersegler";
	loadImagesFromDirectory(Mauersegler_address,mauerseglerTestImages);
	TestLabels.assign(mauerseglerTestImages.size(), 1);
	cout<<""<<mauerseglerTestImages.size()<<"\n";
	cout<<""<<TestLabels;
}
