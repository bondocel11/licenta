using DataUtils;
using Emgu.CV;
using Emgu.CV.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataTraining
{
    public class RBFKernelTraining
    {
        public static SVM Train(double[][] data, int[] outputs)
        {

                SVM model = new SVM();
            
               /* model.C = 1;
                model.Type=SVM.SvmType.CSvc;
                model.SetKernel(SVM.SvmKernelType.Rbf);
                model.Gamma = 0.1;

                var cGrid = new MCvParamGrid();
                cGrid.MinVal = 1e-10;
                cGrid.MaxVal = 2;
                cGrid.Step = 0.05;
                var gammaGrid = new MCvParamGrid();
                gammaGrid.MinVal = 1e-10;
                gammaGrid.MaxVal = 2;
                gammaGrid.Step = 0.05;*/
                var trainData = DoubleToMat(data,outputs);
          
                //bool trained = model.Train(trainData, trainClasses, null, null, p);
                bool trained = model.TrainAuto(trainData, 10);

                return model;
                
                    
        }

        private static TrainData DoubleToMat(double[][] data,int[] outputs)
        {
            Mat matrix = MatUtils.MatrixToMat(data, Emgu.CV.CvEnum.DepthType.Cv32F);
            Mat response = MatUtils.ArrayToMat(outputs, Emgu.CV.CvEnum.DepthType.Cv32S);
            return new TrainData(matrix, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, response);
        }
    }
}
