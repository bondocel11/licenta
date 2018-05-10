using Config;
using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Statistics.Analysis;
using System.Text;
using System.Threading.Tasks;

namespace DiagnosticMeasurement
{
    public static class ComputePerformance
    {
     
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
                        trueNegatives += (Configuration.NR_OF_CLASSES-1) * confusionMatrix[i][j];
                    }

                    else
                    {
                        falseNegatives += confusionMatrix[i][j];
                        trueNegatives += (Configuration.NR_OF_CLASSES-2) * confusionMatrix[i][j];
                    }

                }
                falsePositives = falseNegatives;

            }

            double accuracy = ComputeOverallAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives);
            double specificity = ComputeOverallSpecificity(trueNegatives, falsePositives);
            double recall = ComputeOverallRecall(truePositives, falseNegatives);
            double efficiency = ComputeOverallEfficiency(recall, specificity);
            double precision = ComputeOverallPrecision(truePositives, falsePositives);
            int errors = falsePositives + falseNegatives;
            Console.WriteLine("accuracy " + accuracy);
            Console.WriteLine("efficiency " + efficiency);
            Console.WriteLine("errors " + errors);
            Console.WriteLine("specificity " + specificity);
            Console.WriteLine("recall " + recall);
            Console.WriteLine("precision " + precision);
            Console.WriteLine("TP " + truePositives);
            Console.WriteLine("FP "+ falsePositives);
            Console.WriteLine("TN "+ trueNegatives);
            Console.WriteLine("FN "+ falseNegatives);
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

            double accuracy = ComputeOverallAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives);
            double specificity = ComputeOverallSpecificity(trueNegatives, falsePositives);
            double recall = ComputeOverallRecall(truePositives, falseNegatives);
            double efficiency = ComputeOverallEfficiency(recall, specificity);
            double precision= ComputeOverallPrecision(truePositives, falsePositives);
            int errors = falsePositives + falseNegatives;
       
            Console.WriteLine("accur " + accuracy);
            Console.WriteLine("efficiency " + efficiency);
            Console.WriteLine("errors " + errors);
            Console.WriteLine("specificity " + specificity);
            Console.WriteLine("recall " + recall);
            Console.WriteLine("precision " + precision);

            return accuracy;
        }

        public static int[][] DesignConfusionMatrix(int[] outputs, int[] predicted) { 
           
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

        //verificare corectitudinea calculelor pentru accuracy,specificity,recall,err folosind libraria Accord.
        public static void ComputeGlobalConfusionMatrix(double[][] data, int[] predicted, int[] outputs)
        {
            double accuracy = 0;
            double precision = 0;
            double efficiency = 0;
            double recall = 0;
            double specificity = 0;
            int errors = 0;
            int truePositives = 0;
            int falsePositives = 0;
            int trueNegatives = 0;
            int falseNegatives = 0;
            double fpr = 0;
            for (int j = 0; j < Configuration.NR_OF_CLASSES; j++)
            {
                int[] realOutputs = ReduceValuesForData(outputs, j).ToArray();
                int[] predictedOutputs = ReduceValuesForData(predicted, j).ToArray();
                int positiveValue = 1;
                int negativeValue = 0;
                ConfusionMatrix matrix = new ConfusionMatrix(predictedOutputs, realOutputs, positiveValue, negativeValue);

                accuracy += matrix.Accuracy;
                efficiency += matrix.Efficiency;
                errors += matrix.Errors;
                specificity += matrix.Specificity;
                recall += matrix.Recall;
                precision += matrix.Precision;
                fpr += matrix.FalsePositiveRate;
                truePositives += matrix.TruePositives;
                falsePositives += matrix.FalsePositives;
                trueNegatives += matrix.TrueNegatives;
                falseNegatives += matrix.FalseNegatives;

            }
            double overallSpecificity = ComputeOverallSpecificity(trueNegatives, falsePositives);
            double overallRecall = ComputeOverallRecall(truePositives, falseNegatives);
            Console.WriteLine("Overall accuracy: " + ComputeOverallAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives));
            Console.WriteLine("Overall efficiency: " + ComputeOverallEfficiency(overallRecall, overallSpecificity));
            Console.WriteLine("Overall recall: " + overallRecall);
            Console.WriteLine("Overall specificity: " + overallSpecificity);
            Console.WriteLine("Overall precision: " + ComputeOverallPrecision(truePositives, falsePositives));
            Console.WriteLine("Overall false positive rate: " + (1 - overallSpecificity));
            Console.WriteLine("Total errors: " + errors);
            Console.WriteLine("TP " + truePositives);
            Console.WriteLine("FP " + falsePositives);
            Console.WriteLine("TN " + trueNegatives);
            Console.WriteLine("FN " + falseNegatives);
        }
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
    }
}
