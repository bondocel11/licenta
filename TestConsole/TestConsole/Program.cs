using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using NeuralNet;
using DataPreparation;
using DiagnosticMeasurement;
using System.Drawing;
using DataUtils;
using DataTraining;
using Emgu.CV.ML;
using Emgu.CV.XFeatures2D;
using Emgu.CV.Util;
using ModelValidation;

namespace TestConsole
{
    class Program
    {
       
        static void Main(string[] args)
        {
            DataSets data = new DataSets();
            Mat img = CvInvoke.Imread("E:/Licenta2018/eiffel1.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
           

           // mySIFT.GenerateOctaves(img);
            if (!data.Load(100))
            {
                return;
            }
            double[][] bow=SiftBoWFeatures.ExtractFeatures(data.Train);

            int[] labels=DataUtils.LabelTranslation.TranslateLabels(data.Labels, data.Train.Images.Select(x => x.Label).ToArray());
            var model = RBFKernelTraining.Train(bow, labels);
            double[][] validate= SiftBoWFeatures.ExtractFeatures(data.Validation);
            int[] vallabels = DataUtils.LabelTranslation.TranslateLabels(data.Labels, data.Validation.Images.Select(x => x.Label).ToArray());

            OpencvValidation.Validate(model, validate, vallabels);
            Console.WriteLine("Train with poly ");
            var ovo = PolynomialSVMTraining.Train(bow, labels);
            AccordValidation.Validate(ovo, validate, vallabels);
            Console.WriteLine("Train with Chi ");

            var ovo1 = OtherSVMKernelTraining.Train(bow, labels);
            AccordValidation.Validate(ovo, validate, vallabels);

            /* var tup = SimpleKNeighbors.KNeighborsModel.Train(data);

              int[][] confusionMatrix=ComputePerformance.DesignConfusionMatrix(data.Labels,data.Validation.Images.Select(x => x.Label).ToArray(), SimpleKNeighbors.KNeighborsModel.ClassifyAll(tup,data.Validation.Images));
             Console.WriteLine(SimpleKNeighbors.KNeighborsModel.Classify(tup, img));
             for (int i = 0; i < Config.Configuration.NR_OF_CLASSES; i++)
              {
                  Console.WriteLine("Accuracy of " + i + " " + (float)ComputePerformance.ComputePerformanceMetricsForClass(confusionMatrix, i));

              }
           
            Mat img = CvInvoke.Imread("E:/Licenta2018/eiffel1.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            Mat new_img = new Mat();
            CvInvoke.Resize(img,new_img, new Size(128,128), 0, 0, Inter.Linear);
            double[] hog=DataUtils.DataUtils.HOG(new_img);


            HogSets hs = new HogSets();
            if (!hs.Load(100))
             {
                 return;
             }
            var model= RBFKernelTraining.Train(hs.Train.data, hs.Train.outputs);
           
             //var ovo= GaussianKernelTraining.Train(hs.Train.data, hs.Train.outputs);
            // var ovo1 = OtherSVMKernelTraining.TrainWithOtherSVM(hs.Train.data, hs.Train.outputs);
            */
            Console.ReadKey();

        }
    }
}
