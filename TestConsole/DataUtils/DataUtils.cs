using Config;
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
        public const double PI = Math.PI;

        public static object Config { get; private set; }

        public static int[] calcHist(Mat img)
        {
            int[] histogram = new int[Configuration.NR_BINS * 3];
            for (int i = 0; i < img.Rows; i++)
                for (int j = 0; j < img.Cols; j++)
                {
                    Color pixelColor = img.Bitmap.GetPixel(j, i);
                    histogram[pixelColor.R % Configuration.NR_BINS]++;
                    histogram[pixelColor.G % Configuration.NR_BINS + Configuration.NR_BINS]++;
                    histogram[pixelColor.B % Configuration.NR_BINS + 2 * Configuration.NR_BINS]++;
                }
            return histogram;
        }
        public static double[] HOG(Mat img)
        {

            //this computes the horizontal and vertical gradients
            Mat gx = new Mat();
            Mat gy = new Mat();
            CvInvoke.Sobel(img, gx, Emgu.CV.CvEnum.DepthType.Cv32F, 1, 0, 1);
            CvInvoke.Sobel(img, gy, Emgu.CV.CvEnum.DepthType.Cv32F, 0, 1, 1);


            double[,] mag = ComputeMagnitude(gx, gy);
            double[,] angle = ComputeDirection(gx, gy);

            HOGcell[,] hog_cell = ComputeHogCellHistogram(img, mag, angle);
            return ComputeNormalizedHistogram1(hog_cell);
        }
        private static double[] ComputeNormalizedHistogram1(HOGcell[,] hist_mat)
        {
            int rows = hist_mat.GetLength(0);
            int cols = hist_mat.GetLength(1);
            double[] hist = new double[rows * cols * Configuration.NR_BINS];
            int step = Configuration.STRIDE_SIZE / Configuration.CELL_SIZE;
            int indx = 0;
            double[,] sum = new double[hist_mat.GetLength(0) / step , hist_mat.GetLength(1) / step];
            for (int i = 0; i < rows; i+=step)
            {
                for (int j = 0; j < cols; j += step)
                {
                    for (int ci = 0; ci < step; ci++)
                    {
                        for (int cj = 0; cj < step; cj++)
                        {
                            for (int p = 0; p < Configuration.NR_BINS; p++)
                            {
                                sum[i/step,j/step] += hist_mat[i+ci, j+cj].Histogram[p];
                            }
                           
                        }
                    }
                }
            }
            for (int i = 0; i < rows; i += step)
            {
                for (int j = 0; j < cols; j += step)
                {
                    for (int ci = 0; ci < step; ci++)
                    {
                        for (int cj = 0; cj < step; cj++)
                        {

                            for (int p = 0; p < Configuration.NR_BINS; p++)
                            {
                                if (sum[i/step, j/step] != 0)
                                    hist_mat[i + ci, j + cj].Histogram[p] = hist_mat[i + ci, j + cj].Histogram[p] / sum[i / step, j / step];
                                else hist_mat[i + ci, j + cj].Histogram[p] = 0;
                                hist[indx] = hist_mat[i + ci, j + cj].Histogram[p];
                                indx++;
                            }

                        }
                    }
                }
            }

            return hist;
        }
        private static double[] ComputeNormalizedHistogram(HOGcell[,] hist_mat)
        {
            int rows = hist_mat.GetLength(0);
            int cols = hist_mat.GetLength(1);
 
            double[] hist = new double[rows * cols * Configuration.NR_BINS];
            int indx = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < Configuration.NR_BINS; p++)
                    {
                        sum += hist_mat[i, j].Histogram[p];
                    }
                    for (int p = 0; p < hist_mat[0, 0].Histogram.Count(); p++)
                    {
                        if (sum != 0)
                            hist_mat[i, j].Histogram[p] = hist_mat[i, j].Histogram[p] / sum;
                        else hist_mat[i, j].Histogram[p] = 0;
                        hist[indx] = hist_mat[i, j].Histogram[p];
                        indx++;
                    }
                }
            }
            return hist;
        }

        private static double[][] ComputeCellHistogram(Mat img, double[,] mag, double[,] angle)
        {
            int interval = 180 / Configuration.NR_BINS;
            double[][] hist_mat = new double[(img.Cols / Configuration.CELL_SIZE) * img.Rows / Configuration.CELL_SIZE][];
            int row = 0;
            for (int i = 0; i < angle.GetLength(0); i += Configuration.CELL_SIZE)
            {
                for (int j = 0; j < angle.GetLength(1); j += Configuration.CELL_SIZE)
                {
                    hist_mat[row] = new double[Configuration.NR_BINS];
                    for (int sx = 0; sx < Configuration.CELL_SIZE; sx++)
                    {
                        for (int sy = 0; sy < Configuration.CELL_SIZE; sy++)
                        {
                            int curr_angle = (int)angle[i + sx, j + sy];
                            if (curr_angle <= 160)
                            {
                                hist_mat[row][curr_angle / interval] += mag[i + sx, j + sy];
                            }
                            else
                            {
                                double percentage = (curr_angle - 160) / interval;
                                hist_mat[row][Configuration.NR_BINS - 1] = percentage * mag[i + sx, j + sy];
                                hist_mat[row][0] = (1 - percentage) * mag[i + sx, j + sy];
                            }
                        }
                    }
                    row++;

                }
            }
            return hist_mat;
        }

        private static HOGcell[,] ComputeHogCellHistogram(Mat img, double[,] mag, double[,] angle)
        {
            int interval = 180 / Configuration.NR_BINS;
            HOGcell[,] hist = new HOGcell[img.Rows / Configuration.CELL_SIZE, img.Cols / Configuration.CELL_SIZE];

            for (int i = 0; i < angle.GetLength(0); i += Configuration.CELL_SIZE)
            {
                for (int j = 0; j < angle.GetLength(1); j += Configuration.CELL_SIZE)
                {
                    double[] hist_mat = new double[Configuration.NR_BINS];
                    for (int sx = 0; sx < Configuration.CELL_SIZE; sx++)
                    {
                        for (int sy = 0; sy < Configuration.CELL_SIZE; sy++)
                        {
                            int curr_angle = (int)angle[i + sx, j + sy];
                            if (curr_angle <= 160)
                            {
                                int current_bin = curr_angle / interval;
                                hist_mat[current_bin] += mag[i + sx, j + sy];
                            }
                            else
                            {
                                double percentage = (curr_angle - 160) / (double)interval;
                                hist_mat[Configuration.NR_BINS - 1] += percentage * mag[i + sx, j + sy];
                                hist_mat[0] += (1 - percentage) * mag[i + sx, j + sy];
                            }
                        }
                    }
                    hist[i / Configuration.CELL_SIZE, j / Configuration.CELL_SIZE] = new HOGcell(hist_mat);
                }
            }
            return hist;
        }

        public static double[,] ComputeMagnitude(Mat gx, Mat gy)
        {
            double[,] mag = new double[gx.Rows, gx.Cols];
            for (int i = 0; i < gx.Rows; i++)
            {
                for (int j = 0; j < gy.Cols; j++)
                {
                    double gx_val = gx.GetValue(i, j);
                    double gy_val = gy.GetValue(i, j);
                    mag[i, j] = (double)Math.Round(Math.Sqrt(gx_val * gx_val + gy_val * gy_val), 6);
                }
            }
            return mag;
        }

        public static double[,] ComputeDirection(Mat gx, Mat gy)
        {
            double[,] angle = new double[gx.Rows, gx.Cols];
            for (int i = 0; i < gx.Rows; i++)
            {
                for (int j = 0; j < gy.Cols; j++)
                {
                    double gx_val = gx.GetValue(i, j);
                    double gy_val = gy.GetValue(i, j);
                    angle[i, j] = (double)Math.Round(Math.Abs(Math.Atan2(gy_val, gx_val) * 180 / Math.PI), 4);

                }
            }
            return angle;
        }
    }
}




