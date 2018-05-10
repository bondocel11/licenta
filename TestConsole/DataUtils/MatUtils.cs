using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataUtils
{
    public static class MatUtils
    {
        public static Mat MatrixToMat(double[][] data, Emgu.CV.CvEnum.DepthType dt)
        {

            Mat matrix = new Mat(data.Length, data[0].Length, dt, 1);

            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[0].Length; j++)
                {
                    MatExtensions.SetValue(matrix, i, j, (float)data[i][j]);
                }
            }
            return matrix;
        }

        public static Mat ArrayToMat(int[] outputs, DepthType cv32S)
        {
            Mat response=new Mat(outputs.Length, 1, Emgu.CV.CvEnum.DepthType.Cv32S, 1);
            for (int i = 0; i < outputs.Length; i++)
            {
                MatExtensions.SetValue(response, i, 0, outputs[i]);
            }
            return response;
        }
    }
}
