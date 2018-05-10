using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;

namespace ModelValidation
{
    public class AccordValidation
    {
        public static void Validate(MulticlassSupportVectorMachine<IKernel> model, double[][] data, int[] outputs)
        {
            int[] predicted = model.Decide(data);

            int[][] confMat = DiagnosticMeasurement.ComputePerformance.DesignConfusionMatrix(outputs, predicted);
            DiagnosticMeasurement.ComputePerformance.ComputePerformanceMetrics(confMat);
        }


    }
}
