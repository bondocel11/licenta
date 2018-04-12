using Config;
using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace DataPreparation
{
    public static class DataMolding
    {
       
        internal static Mat ResizeImage(Mat image)
        {
     
                Mat resized_image = new Mat();
                CvInvoke.Resize(image, resized_image, new Size(Configuration.kNewWidth, Configuration.kNewHeight), 0, 0, Inter.Linear);
     
            return resized_image;
        }

        internal static List<Mat> Prepare(List<Mat> dataSet)
        {

            return dataSet.Select(x => ResizeImage(x)).ToList();
        }
    }
}
