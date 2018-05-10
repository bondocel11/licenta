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
    public class OtherSVMKernelTraining
    {
        public static MulticlassSupportVectorMachine<IKernel> Train(double[][] data, int[] outputs)
        {

            var param = GridSearchForSVM(data,outputs);
            return TrainWithSVM(data,outputs, param.Item1, param.Item2);
        }

        private static Tuple<IKernel, List<double>> GridSearchForSVM(double[][] data, int[] outputs)
        {
            //var inputs = data.TrainingData.inputs.Take(218).ToArray();
            //var outputs = data.TrainingData.outputs.Take(218).ToArray();

            var gridsearch = GridSearch<double[], int>.Create(


                ranges: new
                {
                    Kernel = GridSearch.Values<IKernel>(new Linear(), new ChiSquare(), new Sigmoid(),new Gaussian()),
                    Complexity = GridSearch.Range(1e-10, 0.01, stepSize: 0.05),
                    Tolerance = GridSearch.Range(1e-10, 1.0, stepSize: 0.05)
                },
                learner: (p) => new MulticlassSupportVectorLearning<IKernel>()
                {
                    Learner = (parameters) => new SequentialMinimalOptimization<IKernel>()
                    {

                        Kernel = p.Kernel.Value,
                        Tolerance = p.Tolerance,
                        Complexity = p.Complexity

                    }
                },
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),
                loss: (actual, expected, m) => new ZeroOneLoss(expected).Loss(actual)
            );

            gridsearch.ParallelOptions.MaxDegreeOfParallelism = 1;

            var result = gridsearch.Learn(data, outputs);

            var svm = result.BestModel;

            double bestError = result.BestModelError;

            double bestC = result.BestParameters.Complexity.Value;
            double bestTolerance = result.BestParameters.Tolerance.Value;
            var bestKernel = result.BestParameters.Kernel.Value;
            List<double> doubleParam = new List<double>();
            doubleParam.Add(bestC);
            doubleParam.Add(bestTolerance);
            doubleParam.Add(bestError);
            var param = new Tuple<IKernel, List<double>>(bestKernel, doubleParam);
            Console.WriteLine("Kernel used: " + result.BestParameters.Kernel.Value);
            return param;
        }




        public static MulticlassSupportVectorMachine<IKernel> TrainWithSVM(double[][] data, int[] outputs, IKernel kernel, List<double> param)
        {

            var teacher = new MulticlassSupportVectorLearning<IKernel>()
            {
                // Configure the learning algorithm to use SMO to train the
                //  underlying SVMs in each of the binary class subproblems.
                Learner = (p) => new SequentialMinimalOptimization<IKernel>()
                {


                    Kernel = kernel,
                    Tolerance = param[1],
                    Complexity = param[0]
                }

            };

            var ovo = teacher.Learn(data, outputs);

            // Obtain class predictions for each sample
            int[] predicted = ovo.Decide(data);

            // Compute classification error
            double error = new ZeroOneLoss(outputs).Loss(predicted);
       
            return ovo;
        }


    }
}

