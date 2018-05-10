using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Config
{
    public static class Configuration
    {
        public const string urlMnist = @"http://yann.lecun.com/exdb/mnist/";
        public const string mnistFolder = @"..\Mnist\";
        public const string trainingImageFile = "E:/Licenta2018/TrainSet";
        public const string testingLabelFile = "E:/Licenta2018/TrainSet";
        public const string testingImageFile = "E:/Licenta2018/TrainSet";
        public const int kNewWidth = 128;
        public const int kNewHeight = 128;

        //general parameters
        public const int NR_OF_CLASSES = 8;

        //K Neighbors Model
        public const int M = 8;
        public static int K = 20;

        //HOG
        public const int NR_BINS = 9;
        public const int CELL_SIZE = 8;
        public const int STRIDE_SIZE =16;

        //SIFT
        public const int NO_OCTAVES= 4;
        public const int NO_BLUR_LVL = 5;
        public static double K_BLUR = 1.41;
        public static double SIGMA = 0.7;
        public const double CONTRAST_THRESHOLD = 0.03;
    }
}
