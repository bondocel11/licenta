using Accord.MachineLearning.Performance;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;

namespace DataTraining
{
    public static class PolynomialSVMTraining
    {
        public static void TrainWithPolynomialSvm(double[][] data,int[] outputs)
        {

            List<double> param = GridSearchForSVM(data,outputs);
            TrainWithSVM(data, param, outputs);
        }
        public static List<double> GridSearchForSVM(double[][] data, int[] outputs)
        {

            var gscv = GridSearch<double[], int>.CrossValidate(


                    ranges: new
                    {
                        Complexity = GridSearch.Values(0.001, 0.01),
                        Degree = GridSearch.Values(1, 10, 2, 3, 4, 5),
                        Constant = GridSearch.Values(0, 1, 2),
                    },


                    learner: (p, ss) => new MulticlassSupportVectorLearning<Polynomial>
                    {
                        Learner = (parameter) => new SequentialMinimalOptimization<Polynomial>()
                        {
                            Complexity = p.Complexity,
                            Kernel = new Polynomial(p.Degree, p.Constant),

                        }


                    },


                    fit: (teacher, x, y, w) => teacher.Learn(x, y, w),


                    loss: (actual, expected, r) => new ZeroOneLoss(expected).Loss(actual),

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
            double bestConstant = result.BestParameters.Constant;
            Console.WriteLine("C: " + bestC);
            Console.WriteLine("degree: " + bestDegree);
            Console.WriteLine("constant: " + bestConstant);
            List<double> param = new List<double>();
            param.Add(bestC);
            param.Add(bestDegree);
            param.Add(bestError);
            return param;


        }



        public static void TrainWithSVM(double[][] data, List<double> param, int[] outputs)
        {

            var teacher = new MulticlassSupportVectorLearning<Polynomial>()
            {
                // Configure the learning algorithm to use SMO to train the
                //  underlying SVMs in each of the binary class subproblems.
                Learner = (p) => new SequentialMinimalOptimization<Polynomial>()
                {

                    Complexity = param[0],
                    Kernel = new Polynomial((int)param[1], param[2])

                }

            };

            MulticlassSupportVectorMachine<Polynomial> ovo = teacher.Learn(data, outputs);

            // Obtain class predictions for each sample
            int[] predicted = ovo.Decide(data);

            // Compute classification error
            double error = new ZeroOneLoss(outputs).Loss(predicted);
            Console.WriteLine("Polynomial SVM");
            DiagnosticMeasurement.ComputePerformance.ComputeGlobalConfusionMatrix(data, predicted,outputs);

        }


      
    }
}