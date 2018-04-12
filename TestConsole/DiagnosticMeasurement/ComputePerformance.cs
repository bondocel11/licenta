using Config;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiagnosticMeasurement
{
    public static class ComputePerformance
    {
        
        private static double ComputeOverallAccuracy(int truePositives, int trueNegatives, int falsePositives, int falseNegatives)
        {
            return (double)(truePositives + trueNegatives) / (trueNegatives + truePositives + falseNegatives + falsePositives);
        }

        private static double ComputeOverallPrecision(int truePositives, int falsePositives)
        {
            return (double)truePositives / (truePositives + falsePositives);
        }

        private static double ComputeOverallEfficiency(double overallRecall, double overallSpecificity)
        {
            return (double)(overallSpecificity + overallRecall) / 2;
        }

        private static double ComputeOverallSpecificity(int trueNegatives, int falsePositives)
        {
            return (double)trueNegatives / (trueNegatives + falsePositives);
        }

        private static double ComputeOverallRecall(int truePositives, int falseNegatives)
        {
            return (double)truePositives / (truePositives + falseNegatives);
        }


        private static List<int> ReduceValuesForData(int[] data, int j)
        {
            List<int> realOutputs = new List<int>();
            data.ToList().ForEach(i =>
            {
                if (i == j)
                {
                    realOutputs.Add(1);
                }
                else realOutputs.Add(0);
            });

            return realOutputs;
        }

        public static double ComputePerformanceMetrics(int[][] confusionMatrix)
        {
            int truePositives = 0;
            int trueNegatives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;
            for (int i = 0; i < Configuration.NR_OF_CLASSES; i++)
            {

                for (int j = 0; j < Configuration.NR_OF_CLASSES; j++)
                {
                    if (i == j)
                    {
                        truePositives += confusionMatrix[i][j];
                        trueNegatives += 4 * confusionMatrix[i][j];
                    }

                    else
                    {
                        falseNegatives += confusionMatrix[i][j];
                        trueNegatives += 3 * confusionMatrix[i][j];
                    }

                }
                falsePositives = falseNegatives;

            }
            double accuracy = ComputeOverallAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives);
            Console.WriteLine("Overall performance");
            int noOfEg = truePositives + falseNegatives;
            Console.WriteLine("Number of test examples " + noOfEg);
            double overallSpecificity = ComputeOverallSpecificity(trueNegatives, falsePositives);
            double overallRecall = ComputeOverallRecall(truePositives, falseNegatives);

            Console.WriteLine("Overall accuracy: " + accuracy);
            Console.WriteLine("Overall efficiency: " + ComputeOverallEfficiency(overallRecall, overallSpecificity));
            Console.WriteLine("Overall recall: " + overallRecall);
            Console.WriteLine("Overall specificity: " + overallSpecificity);
            Console.WriteLine("Overall precision: " + ComputeOverallPrecision(truePositives, falsePositives));
            Console.WriteLine("Overall false positive rate: " + (1 - overallSpecificity));
            // Console.WriteLine("Total errors: " + errors);
            Console.WriteLine("\n\n\n");
            return accuracy;
        }
        public static double ComputePerformanceMetricsForClass(int[][] confusionMatrix, int actual)
        {
            int truePositives = confusionMatrix[actual][actual];
            int trueNegatives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;

            for (int predicted = 0; predicted < Configuration.NR_OF_CLASSES; predicted++)
            {
                if (predicted != actual)
                {
                    falsePositives += confusionMatrix[actual][predicted];
                    falseNegatives += confusionMatrix[predicted][actual];

                }
                for (int j = 0; j < Configuration.NR_OF_CLASSES; j++)
                {
                    if ((j != actual) && (predicted != actual))
                    {
                        trueNegatives += confusionMatrix[predicted][j];
                    }

                }

            }

            Console.WriteLine("Confusion matrix for class " + actual);
            int noOfEg = truePositives + falseNegatives;
            Console.WriteLine("Number of test examples " + noOfEg);
            double accuracy = ComputeOverallAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives);
            double overallSpecificity = ComputeOverallSpecificity(trueNegatives, falsePositives);
            double overallRecall = ComputeOverallRecall(truePositives, falseNegatives);

            Console.WriteLine("Overall accuracy: " + accuracy);
            Console.WriteLine("Overall efficiency: " + ComputeOverallEfficiency(overallRecall, overallSpecificity));
            Console.WriteLine("Overall recall: " + overallRecall);
            Console.WriteLine("Overall specificity: " + overallSpecificity);
            Console.WriteLine("Overall precision: " + ComputeOverallPrecision(truePositives, falsePositives));
            Console.WriteLine("Overall false positive rate: " + (1 - overallSpecificity));
            // Console.WriteLine("Total errors: " + errors);
            Console.WriteLine("\n");
            return accuracy;
        }

        public static int[][] DesignConfusionMatrix(List<string> allLabels, string[] outputLabels, string[] predictedLabels)
        {
            int[] outputs = TranslateLabels(allLabels,outputLabels);
            int[] predicted = TranslateLabels(allLabels,predictedLabels);

            int[][] confusionMatrix = new int[Configuration.NR_OF_CLASSES][];
            for (int i = 0; i < Configuration.NR_OF_CLASSES; i++)
            {
                confusionMatrix[i] = new int[Configuration.NR_OF_CLASSES];

            }
            for (int i = 0; i < outputs.Length; i++)
            {
                int actual = outputs[i];
                int predict = predicted[i];
                confusionMatrix[predict][actual]++;
            }
            return confusionMatrix;
        }

        private static int[] TranslateLabels(List<string> allLabels, string[] testLabels)
        {

            return testLabels.Select(label => allLabels.IndexOf(label)).ToArray();
        }
    }
}
