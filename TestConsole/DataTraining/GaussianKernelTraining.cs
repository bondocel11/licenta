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
    public  class GaussianKernelTraining
    {
        public static MulticlassSupportVectorMachine<Gaussian> Train(double[][] data, int[] outputs)
        {

           // var param = GridSearchForSVM(data, outputs);
            return TrainWithSVM(data, outputs);
        }


        public static MulticlassSupportVectorMachine<Gaussian> TrainWithSVM(double[][] data, int[] outputs)
        {

            var teacher = new MulticlassSupportVectorLearning<Gaussian>()
            {
                // Configure the learning algorithm to use SMO to train the
                //  underlying SVMs in each of the binary class subproblems.
                Learner = (p) => new SequentialMinimalOptimization<Gaussian>()
                {
                    
                    UseKernelEstimation = true
                }

            };

            var machine= teacher.Learn(data, outputs);



            // Create the multi-class learning algorithm for the machine
            var calibration = new MulticlassSupportVectorLearning<Gaussian>()
            {
                Model = machine, // We will start with an existing machine

                // Configure the learning algorithm to use Platt's calibration
                Learner = (param) => new ProbabilisticOutputCalibration<Gaussian>()
                {
                    Model = param.Model // Start with an existing machine
                }
            };


            // Configure parallel execution options
            calibration.ParallelOptions.MaxDegreeOfParallelism = 1;

            // Learn a machine
            calibration.Learn(data, outputs);

            // Obtain class predictions for each sample
            int[] predicted = machine.Decide(data);

            // Get class scores for each sample
            double[] scores = machine.Score(data);

            // Get log-likelihoods (should be same as scores)
            double[][] logl = machine.LogLikelihoods(data);

            // Get probability for each sample
            double[][] prob = machine.Probabilities(data);
            // Obtain class predictions for each sample
          
            // Compute classification error
            double error = new ZeroOneLoss(outputs).Loss(predicted);

            return machine;
        }
    }
}
