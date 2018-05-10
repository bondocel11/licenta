using DataUtils;
using Emgu.CV;
using Emgu.CV.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelValidation
{
    public static class OpencvValidation
    {
        public static void Validate(SVM model,double[][] data, int[] outputs)
        {
            Mat matrix = MatUtils.MatrixToMat(data, Emgu.CV.CvEnum.DepthType.Cv32F);
            Mat results = new Mat(data.Length, 1, Emgu.CV.CvEnum.DepthType.Cv32S, 1);

            model.Predict(matrix, results);
            int[] predicted = new int[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                predicted[i] = (int)results.GetValue(i, 0);
            }
            int[][] confMat = DiagnosticMeasurement.ComputePerformance.DesignConfusionMatrix(outputs, predicted);
            DiagnosticMeasurement.ComputePerformance.ComputePerformanceMetrics(confMat);

            Console.WriteLine("Kernel: " + model.KernelType);
            Console.WriteLine("C : " + model.C);
            Console.WriteLine("Kernel: " + model.Gamma); 
        }
    }
}
