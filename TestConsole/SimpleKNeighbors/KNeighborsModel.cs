using DataLoader;
using DataPreparation;
using Emgu.CV;
using System;
using System.Linq;
using System.Collections.Generic;
using Config;

namespace SimpleKNeighbors
{

    public static class KNeighborsModel
    {
        
        public static Tuple<double[,],string[]> Train(DataSets trainData)
        {

            double[,]  X = new double[trainData.Train.Images.Count(),3 * Configuration.M];
            string[] Y = new string[trainData.Train.Images.Count()];
            int i = 0;
            trainData.Train.Images.ForEach(inputEntry =>
            {
                int[] histogram = DataUtils.DataUtils.calcHist(inputEntry.Image, Configuration.M);
                for (int d = 0; d < histogram.Length; d++)
                    X[i, d]= histogram[d];
                Y[i] = inputEntry.Label;
                i++;
            });
            
            return Tuple.Create(X,Y);
        }

        private static string Classify(Tuple<double[,], string[]> dataset, Mat unknown)
        {
            int[] histogram = DataUtils.DataUtils.calcHist(unknown, Configuration.M);
            int j = 0;
            Distance[] distances = new Distance[dataset.Item1.GetLength(0)];
            for (int i = 0; i < dataset.Item1.GetLength(0); i++)
            {

                double dist=histogram.Select(x => Math.Sqrt(Math.Abs(x * x - dataset.Item1[i,j] * dataset.Item1[i,j]))).Sum();
                j++;
                if (j == 24) j = 0;
                distances[i] = new Distance(dist, dataset.Item2[i]);

            }
            Distance[] sortedData = distances.OrderBy(x => x.Dist).ToArray();
            return Vote(sortedData);
        }

        private static string Vote(Distance[] sortedData)
        {
            return sortedData.Take(Configuration.K).GroupBy(distance => distance.Label).OrderByDescending(g => g.Count()).SelectMany(q => q).ElementAt(0).Label;
        }

        public static string[] ClassifyAll(Tuple<double[,], string[]> dataset, List<InputEntry> images)
        {
            return images.Select(image => Classify(dataset, image.Image)).ToArray();
        }

        private class Distance
        {

            private double dist;
            private string label;

            public Distance(double dist, string label)
            {
                this.Dist = dist;
                this.Label = label;
            }

            public double Dist { get => dist; set => dist = value; }
            public string Label { get => label; set => label = value; }
        }
    }
}
