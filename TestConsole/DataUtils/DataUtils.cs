using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataUtils
{
    public static class DataUtils
    {
        public static int[] calcHist(Mat img, int nr_bins)
        {
            int[] histogram = new int[nr_bins * 3];
            for (int i=0;i<img.Rows;i++)
                for (int j = 0; j < img.Cols; j++)
                {
                    Color pixelColor = img.Bitmap.GetPixel(j,i);
                    histogram[pixelColor.R % nr_bins]++;
                    histogram[pixelColor.G % nr_bins + nr_bins]++;
                    histogram[pixelColor.B % nr_bins + 2*nr_bins]++;
                }
            return histogram;
        }
    }
}
