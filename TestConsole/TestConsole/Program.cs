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

namespace TestConsole
{
    class Program
    {
       
        static void Main(string[] args)
        {
            /* String win1 = "Test Window"; //The name of the window
             CvInvoke.NamedWindow(win1); //Create the window using the specific name

             Mat img = new Mat(200, 400, DepthType.Cv8U, 3); //Create a 3 channel image of 400x200
             img.SetTo(new Bgr(255, 0, 0).MCvScalar); // set it to Blue color

             //Draw "Hello, world." on the image using the specific font
             CvInvoke.PutText(
                img,
                "Hello, world",
                new System.Drawing.Point(10, 80),
                FontFace.HersheyComplex,
                1.0,
                new Bgr(0, 255, 0).MCvScalar);


             CvInvoke.Imshow(win1, img); //Show the image
             CvInvoke.WaitKey(0);  //Wait for the key pressing event
             CvInvoke.DestroyWindow(win1); //Destroy the window if key is pressed*/
            DataSets data = new DataSets();
            if (!data.Load(100))
            {
                return;
            }
            var tup = SimpleKNeighbors.KNeighborsModel.Train(data);
            Mat img = CvInvoke.Imread("D:/licenta/eiffel1.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            int[][] confusionMatrix=ComputePerformance.DesignConfusionMatrix(data.Labels,data.Validation.Images.Select(x => x.Label).ToArray(), SimpleKNeighbors.KNeighborsModel.ClassifyAll(tup,data.Validation.Images));
            //Console.WriteLine(SimpleKNeighbors.KNeighborsModel.Classify(tup, img));
            for (int i = 0; i < Config.Configuration.NR_OF_CLASSES; i++)
            {
                Console.WriteLine("Accuracy of " + i + " " + (float)ComputePerformance.ComputePerformanceMetricsForClass(confusionMatrix, i));

            }
            
            Console.ReadKey();
        }
    }
}
