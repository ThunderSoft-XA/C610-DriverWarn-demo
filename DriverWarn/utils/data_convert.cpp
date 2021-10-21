#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace cv;

/**
 * provide serial function for converting tf tensorflow type to cv Mat,or reverse. 
 * **/
void cvMat2tfTensor(cv::Mat input, float inputMean, float inputSTD, tensorflow::Tensor& outputTensor)
{
    auto outputTensorMapped = outputTensor.tensor<float, 4>();

    input.convertTo(input, cv::CV_32FC3);
    cv::resize(input, input, cv::Size(inputWidth, inputHeight));

    input = input - inputMean;
    input = input / inputSTD;

    int height = input.size().height;
    int width = input.size().width;
    int depth = input.channels();

    const float* data = (float*)input.data;
    for (int y = 0; y < height; ++y)
    {
        const float* dataRow = data + (y * width * depth);
        for (int x = 0; x < width; ++x)
        {
            const float* dataPixel = dataRow + (x * depth);
            for (int c = 0; c < depth; ++c)
            {
                const float* dataValue = dataPixel + c;
                outputTensorMapped(0, y, x, c) = *dataValue;
            }
        }
    }
}

void cvMat2Tensor(cv::Mat &image, Tensor &t) 
{
   resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE)); //对图片进行缩放
   auto output = t.shaped<float, 4>({ 1, IMAGE_SIZE, IMAGE_SIZE, 3});
   for (int i = 0; i < image.rows; ++i) {
       for (int j = 0; j < image.cols; ++j) {
           for (int k = 0; k < 3; ++k) {
               output(0, i, j, k) = image.at<Vec3b>(i, j)[2-k];    
           }
       }
    }
}

void cvMat2Tensor(Mat &image, Tensor &t)
{
    cvtColor(image, image, cv::COLOR_BGR2RGB);      
    float *tensor_data_ptr = t.flat<float>().data();              
    cv::Mat fake_mat(image.rows, image.cols, CV_32FC(image.channels()), tensor_data_ptr); 
    image.convertTo(fake_mat, CV_32FC(image.channels()));
}


int tfTensor2cvMat(const tensorflow::Tensor& inputTensor, cv::Mat& output)
{
    tensorflow::TensorShape inputTensorShape = inputTensor.shape();
    if (inputTensorShape.dims() != 4)
    {
        return -1;
    }

    int height = inputTensorShape.dim_size(1);
    int width = inputTensorShape.dim_size(2);
    int depth = inputTensorShape.dim_size(3);

    output = cv::Mat(height, width, CV_32FC(depth));
    auto inputTensorMapped = inputTensor.tensor<float, 4>();
    float* data = (float*)output.data;
    for (int y = 0; y < height; ++y)
    {
        float* dataRow = data + (y * width * depth);
        for (int x = 0; x < width; ++x)
        {
            float* dataPixel = dataRow + (x * depth);
            for (int c = 0; c < depth; ++c)
            {
                float* dataValue = dataPixel + c;
                *dataValue = inputTensorMapped(0, y, x, c);
            }
        }
    }
    return 0;
}

void tfTensor2cvMat(Tensor &t, Mat &image) {
    image.convertTo(image, CV_32FC1);
    tensorflow::StringPiece tmp_data = t.tensor_data();
    memcpy(image.data,const_cast<char*>(tmp_data.data()),IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    image.convertTo(image, CV_8UC1);
}

void tfTensor2cvMat(Tensor &t, Mat &image) {
    int *p = t.flat<int>().data();
    image = Mat(IMAGE_SIZE, IMAGE_SIZE, CV_32SC1, p);
    image.convertTo(image, CV_8UC1);
}

// end of convert function