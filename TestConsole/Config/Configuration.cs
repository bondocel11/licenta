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
        public const string trainingImageFile = "D:/licenta/TrainSet";
        public const string testingLabelFile = "D:/licenta/TrainSet";
        public const string testingImageFile = "D:/licenta/TrainSet";
        public const int kNewWidth = 128;
        public const int kNewHeight = 128;

        //general parameters
        public const int NR_OF_CLASSES = 8;

        //K Neighbors Model
        public const int M = 8;
        public static int K = 20;
    }
}
