using DataUtils;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPreparation
{
    public static class SiftBoWFeatures
    {
        public static double[][] ExtractFeatures(DataSet data)
        {
            List<Mat> allDescriptors = new List<Mat>();
            Mat sth = new Mat();
            data.Images.ForEach(x =>
            {
                SIFT sf = new SIFT(0, 4, 0.03, 10, 1.6);
                VectorOfKeyPoint vectorOfKeyPoints = new VectorOfKeyPoint();
                Mat descriptor = new Mat();
                sf.DetectAndCompute(x.Image, null, vectorOfKeyPoints, descriptor, false);
                allDescriptors.Add(descriptor);
            });

            return ConstructBoWKMeans(allDescriptors,data);
        }

        private static double[][] ConstructBoWKMeans(List<Mat> allDescriptors,DataSet data)
        {
            int dictionarySize = 900;
            MCvTermCriteria tm = new MCvTermCriteria(100, 0.001);
            BOWKMeansTrainer bow = new BOWKMeansTrainer(dictionarySize, tm, 1, Emgu.CV.CvEnum.KMeansInitType.PPCenters);
            allDescriptors.ForEach(x => bow.Add(x));
            Mat dictionary = new Mat();
            bow.Cluster(dictionary);
            SIFT sf = new SIFT(0, 4, 0.03, 10, 1.6);
            double[][] descriptors = new double[data.Images.Count()][];
            data.Images.ForEach(x =>
            {
                descriptors[data.Images.IndexOf(x)] = new double[dictionarySize];
                MKeyPoint[] keyPoints= sf.Detect(x.Image);
                Mat descriptor = new Mat();
                //FlannBasedMatcher matcher = new FlannBasedMatcher(new KMeansIndexParams(), new SearchParams());
                BFMatcher matcher = new BFMatcher(DistanceType.L2, false);
                VectorOfKeyPoint vectorOfKeyPoints = new VectorOfKeyPoint(keyPoints);
                BOWImgDescriptorExtractor bOWImgDescriptorExtractor = new BOWImgDescriptorExtractor(sf,matcher);
                bOWImgDescriptorExtractor.SetVocabulary(dictionary);
                bOWImgDescriptorExtractor.Compute(x.Image, vectorOfKeyPoints, descriptor);
                for (int i = 0; i < dictionarySize; i++)
                {
                   
                    descriptors[data.Images.IndexOf(x)][i]= (double)descriptor.GetValue(0, i);
                    
                }
               
               
            });

            return descriptors;

        }
    }
}
