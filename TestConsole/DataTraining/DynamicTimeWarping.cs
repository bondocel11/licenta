using Accord.MachineLearning.Performance;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataTraining
{
    public class DynamicTimeWarping
    {
        public static MulticlassSupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping> Train(double[][] data, int[] outputs)
        {

            //List<double> param = GridSearchForSVM(data, outputs);
      //      return TrainWithSVM(data, param, outputs);
            return TrainWithSVM1(data, outputs);
        }
        public static List<double> GridSearchForSVM(double[][] data, int[] outputs)
        {
            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < data[0].GetLength(0); j++)
                {
                    if (Double.IsNaN(data[i][j]))
                    {
                        ;
                    }
                }
            }
            var gscv = GridSearch<double[], int>.CrossValidate(

                    ranges: new
                    {
                        Complexity = global::Accord.MachineLearning.Performance.GridSearch.Values<double>(1e-3),
                        Degree = global::Accord.MachineLearning.Performance.GridSearch.Values<int>(1),
                        Alpha = global::Accord.MachineLearning.Performance.GridSearch.Values<int>(0),
                    },


                    learner: (p, ss) => new MulticlassSupportVectorLearning<Accord.Statistics.Kernels.DynamicTimeWarping>
                    {
                        Learner = (Accord.MachineLearning.InnerParameters<SupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping>, double[]> parameter) => new SequentialMinimalOptimization<Accord.Statistics.Kernels.DynamicTimeWarping>()
                        {
                            Complexity = p.Complexity,
                            Kernel = new Accord.Statistics.Kernels.DynamicTimeWarping(alpha: (int)p.Alpha, degree: (int)p.Degree)

                        }

                    },


                    fit: (MulticlassSupportVectorLearning<Accord.Statistics.Kernels.DynamicTimeWarping> teacher, double[][] x, int[] y, double[] w) => teacher.Learn(x, y, w),


                    loss: (int[] actual, int[] expected, MulticlassSupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping> r) => new ZeroOneLoss(expected).Loss(actual),

                    folds: 3 // use k = 3 in k-fold cross validation
            );

            // If needed, control the parallelization degree
            gscv.ParallelOptions.MaxDegreeOfParallelism = 1;

            // Search for the best vector machine
            var result = gscv.Learn(data, outputs);

            // Get the best cross-validation result:
            var crossValidation = result.BestModel;

            // Estimate its error:
            double bestError = result.BestModelError;
            double trainError = result.BestModel.Training.Mean;
            double trainErrorVar = result.BestModel.Training.Variance;
            double valError = result.BestModel.Validation.Mean;
            double valErrorVar = result.BestModel.Validation.Variance;

            // Get the best values for the parameters:
            double bestC = result.BestParameters.Complexity;
            double bestDegree = result.BestParameters.Degree;
            double bestAlpha = result.BestParameters.Alpha;
            Console.WriteLine("C: " + bestC);
            Console.WriteLine("degree: " + bestDegree);
            Console.WriteLine("alpha: " + bestAlpha);
            List<double> param = new List<double>();
            param.Add(bestC);
            param.Add(bestDegree);
            param.Add(bestAlpha);
            return param;


        }



        public static MulticlassSupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping> TrainWithSVM1(double[][] data, int[] outputs)
        {

            var teacher = new MulticlassSupportVectorLearning<Accord.Statistics.Kernels.DynamicTimeWarping>()
            {
                // Configure the learning algorithm to use SMO to train the
                //  underlying SVMs in each of the binary class subproblems.
                Learner = (Accord.MachineLearning.InnerParameters<SupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping>, double[]> p) => new SequentialMinimalOptimization<Accord.Statistics.Kernels.DynamicTimeWarping>()
                {


                    Complexity = 0.001,
                    Kernel = new Accord.Statistics.Kernels.DynamicTimeWarping(alpha: 0, degree: 1)

                }

            };

            MulticlassSupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping> ovo = teacher.Learn(data, outputs);

            // Obtain class predictions for each sample
            int[] predicted = ovo.Decide(data);

            // Compute classification error
            double error = new ZeroOneLoss(outputs).Loss(predicted);
            return ovo;
        }
        public static MulticlassSupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping> TrainWithSVM(double[][] data, List<double> param, int[] outputs)
        {

            var teacher = new MulticlassSupportVectorLearning<Accord.Statistics.Kernels.DynamicTimeWarping>()
            {
                // Configure the learning algorithm to use SMO to train the
                //  underlying SVMs in each of the binary class subproblems.
                Learner = (Accord.MachineLearning.InnerParameters<SupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping>, double[]> p) => new SequentialMinimalOptimization<Accord.Statistics.Kernels.DynamicTimeWarping>()
                {


                    Complexity = 0.001,
                    Kernel = new Accord.Statistics.Kernels.DynamicTimeWarping(alpha: 0, degree: 1)

                }

            };

            MulticlassSupportVectorMachine<Accord.Statistics.Kernels.DynamicTimeWarping> ovo = teacher.Learn(data, outputs);

            // Obtain class predictions for each sample
            int[] predicted = ovo.Decide(data);

            // Compute classification error
            double error = new ZeroOneLoss(outputs).Loss(predicted);
            return ovo;
        }
    }
}
