//
//  main.cpp
//  SortImage
//
//  Created by jiongQiao on 16/1/26.
//  Copyright © 2016年 jiongQiao. All rights reserved.
//


#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
using namespace cv;
using namespace ml;
using namespace std;


int main(int argc, char** argv)
{
    int ImgWidht = 32;
    int ImgHeight = 32;
#define 我懒得改了反正就两句 HOGDescriptor *hog=new HOGDescriptor(cvSize(ImgWidht,ImgHeight),cvSize(16,16),cvSize(8,8),cvSize(16,16), 9);
    vector<string> img_path;
    vector<int> img_catg;
    int nLine = 0;
    string buf;
    //ifstream svm_data( "/Volumes/RamDisk/SVM_DATA.txt" );
    ifstream svm_text("/Volumes/Data/text.txt");
    ifstream svm_non_text("/Volumes/Data/non-text.txt");
    unsigned long n;
    clock_t timeStart=clock();
    cout<<"Program start"<<endl;
    //检测样本
    vector<string> img_tst_path;
    vector<int> img_tst_catg;
//    ifstream img_tst( "/Volumes/RamDisk/SVM_TEST.txt" );
//    while( img_tst )
//    {
//        if( getline( img_tst, buf ) )
//        {
//            img_tst_path.push_back( buf );
//        }
//    }
//    img_tst.close();
    
    
//    while( svm_data )//将训练样本文件依次读取进来
//    {
//        if( getline( svm_data, buf ) )
//        {
//            if(nLine%2==0&&rand()%1048576-3334<2)img_tst_path.push_back(buf.c_str());
//            if((rand()%65536)<1300){
//                nLine++;
//            if( nLine % 2 == 0 )//这里的分类比较有意思，看得出来上面的SVM_DATA.txt文本中应该是一行是文件路径，接着下一行就是该图片的类别，可以设置为0或者1，当然多个也无所谓
//            {
//                img_catg.push_back( atoi( buf.c_str() ) );//atoi将字符串转换成整型，标志（0,1），注意这里至少要有两个类别，否则会出错
//            }
//            else
//            {
//                img_path.push_back( buf );//图像路径
//            }}
//            else{
//                getline( svm_data, buf );
//            }
//        }
//    }
    //svm_data.close();//关闭文件
    
    for(int i=0;i<5000;i++)//将训练样本文件依次读取进来
    {
        if( getline( svm_text, buf ))
        {
            
            if(i%10){
                nLine+=2;
                img_catg.push_back( 1 );//atoi将字符串转换成整型，标志（0,1），注意这里至少要有两个类别，否则会出错
                img_path.push_back( buf );//图像路径
            }
            else{
                img_tst_path.push_back( buf );
                img_tst_catg.push_back( 1 );
            }
            
        }
    }
//    for(int i=0;i<50;i++){
//        getline( svm_text, buf );
//        img_tst_path.push_back( buf );
//    }
    for(int i=0;i<5000;i++)//将训练样本文件依次读取进来
    {
        if( getline( svm_non_text, buf ) )
        {
            
            if(i%10){
                nLine+=2;
                img_catg.push_back( 0 );//atoi将字符串转换成整型，标志（0,1），注意这里至少要有两个类别，否则会出错
                img_path.push_back( buf );//图像路径
            }
            else{
                img_tst_path.push_back( buf );
                img_tst_catg.push_back( 0 );
            }
        }
    }
    
//    for(int i=0;i<50;i++){
//        getline( svm_non_text, buf );
//        img_tst_path.push_back( buf );
//    }
    
    
    Mat data_mat, res_mat;
    int nImgNum = nLine;            //读入样本数量
    ////样本矩阵，nImgNum：横坐标是样本数量， WIDTH * HEIGHT：样本特征向量，即图像大小
    //data_mat = Mat::zeros( nImgNum, 12996, CV_32FC1 );
    //类型矩阵,存储每个样本的类型标志
    res_mat = Mat::zeros( nImgNum, 1, CV_32S );
    
    Mat src;
    Mat trainImg = Mat::zeros(ImgHeight, ImgWidht, CV_8UC1);//需要分析的图片
    
    for( string::size_type i = 0; i != img_path.size(); i++ )
    {
        src = imread(img_path[i].c_str(), 1);
        
        //cout<<" processing "<<img_path[i].c_str()<<endl;
        
        resize(src, trainImg, cv::Size(ImgWidht,ImgHeight), 0, 0, INTER_CUBIC);
        我懒得改了反正就两句  //具体意思见参考文章1,2
        vector<float>descriptors;//结果数组
        hog->compute(trainImg, descriptors, Size(1,1), Size(0,0)); //调用计算函数开始计算
        if (i==0)
        {
            data_mat = Mat::zeros( nImgNum, descriptors.size(), CV_32FC1 ); //根据输入图片大小进行分配空间
        }
        if(i==0)cout<<"HOG dims: "<<descriptors.size()<<endl;
        n=0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            data_mat.at<float>(i,n) = *iter;
            n++;
        }
        //cout<<SVMtrainMat->rows<<endl;
        res_mat.at<float>(i, 0) =  img_catg[i];
        //cout<<img_catg[i]<<endl;
        //cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;
    }
    
    clock_t timeProcess=clock();
    cout<<"Process complete in "<< (double)(timeProcess-timeStart)/CLOCKS_PER_SEC<<" seconds"<<endl;
    Ptr<SVM> svm=SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::EPS, 200, FLT_EPSILON));
    
//    CvSVM svm = CvSVM();
//    CvSVMParams param;
//    CvTermCriteria criteria;
//    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
//    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );
    
    /*
     SVM种类：CvSVM::C_SVC
     Kernel的种类：CvSVM::RBF
     degree：10.0（此次不使用）
     gamma：8.0
     coef0：1.0（此次不使用）
     C：10.0
     nu：0.5（此次不使用）
     p：0.1（此次不使用）
     然后对训练数据正规化处理，并放在CvMat型的数组里。
     */
    //☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆
    svm->train( data_mat,SampleTypes::ROW_SAMPLE, res_mat);
    //☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆
    svm->save( "SVM_DATA2.xml" );
    
    clock_t timeTrain=clock();
    cout<<"Train complete in "<< (double)(timeTrain-timeProcess)/CLOCKS_PER_SEC<<" seconds"<<endl;
    
    //svm->load<SVM>("SVM_DATA.xml");
    
    Mat test;
    char line[512];
    ofstream predict_txt( "/Volumes/Data/SVM_PREDICT.txt" );
    int predsucc1=0,predsucc0=0;
    string::size_type j;
    for( j = 0; j != img_tst_path.size(); j++ )
    {
        //test=imread(img_tst_path[j]);
        test = imread( img_tst_path[j]);//读入图像
        resize(test, trainImg, cv::Size(ImgWidht,ImgHeight), 0, 0, INTER_CUBIC);//要搞成同样的大小才可以检测到
        我懒得改了反正就两句  //具体意思见参考文章1,2
        vector<float>descriptors;//结果数组
        hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算
        //cout<<"The Detection Result:"<<endl;
        //cout<<"HOG dims: "<<descriptors.size()<<endl;
        Mat SVMtrainMat =  Mat::zeros(1,descriptors.size(),CV_32FC1);
        n=0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            SVMtrainMat.at<float>(0,n) = *iter;
            n++;
        }
        
        int ret = svm->predict(SVMtrainMat);
        if(ret&&img_tst_catg[j]){
            predsucc1++;
        }
        else if(!ret&&!img_tst_catg[j]){
            predsucc0++;
        }
        std::sprintf( line, "%s %d\r\n", img_tst_path[j].c_str(), ret );
        //printf("%s %d\r\n", img_tst_path[j].c_str(), ret);
        //getchar();
        predict_txt<<line;  
    }
    clock_t timePredict=clock();
    cout<<"Predict complete in "<< (double)(timePredict-timeTrain)/CLOCKS_PER_SEC<<" seconds"<<endl;
    printf("%f%% %f%%\n",(double)predsucc1*200/j,(double)predsucc0*200/j);
    predict_txt.close();  
    
    return 0;  
}