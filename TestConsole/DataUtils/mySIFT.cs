using DataUtils;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataUtils
{
    public static class mySIFT
    {
        //http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/
        public static void GenerateOctaves(Mat img)
        {
            //aici intrebare e: E bine luat k si sigma?
            Mat gray = img.Clone();
            if (img.NumberOfChannels==3 || img.NumberOfChannels==4)
            {
               CvInvoke.CvtColor(img, gray, ColorConversion.Bgr2Gray);
            }
            Octave[] octaves = new Octave[Config.Configuration.NO_OCTAVES];
            Mat resized = new Mat();
            double[,] sigmas=new double[Config.Configuration.NO_OCTAVES, Config.Configuration.NO_BLUR_LVL];
            for (int i=0;i<Config.Configuration.NO_OCTAVES;i++)
            {
                double prev_sigma = Math.Pow(2,i) * Config.Configuration.SIGMA;
                octaves[i] = new Octave();
                octaves[i].images[0] = new Mat();
                if (i == 0)
                {
                    CvInvoke.Resize(gray, octaves[i].images[0], new System.Drawing.Size(img.Rows * 2, img.Cols * 2));
                }
                else
                {
                    CvInvoke.Resize(gray, octaves[i].images[0], new System.Drawing.Size(octaves[i - 1].images[0].Rows / 2, octaves[i - 1].images[0].Cols / 2));

                }
                for (int j = 0; j < Config.Configuration.NO_BLUR_LVL; j++)
                {
                    if (j == 0)
                    {
                        
                        CvInvoke.GaussianBlur(octaves[i].images[0], octaves[i].images[0], new System.Drawing.Size(13, 13), prev_sigma);
                        sigmas[i, j] = prev_sigma;
                        //CvInvoke.Imshow("grey", octaves[i].images[0]);
                        //CvInvoke.WaitKey();
                    }
                    else
                    {
                        double new_sigma = prev_sigma * Config.Configuration.K_BLUR;
                        octaves[i].images[j] = new Mat();
                        sigmas[i, j] = new_sigma;
                        CvInvoke.GaussianBlur(octaves[i].images[j - 1], octaves[i].images[j], new System.Drawing.Size(13, 13), new_sigma);
                        prev_sigma = new_sigma;
                        //CvInvoke.Imshow("grey", octaves[i].images[j]);
                        //CvInvoke.WaitKey();
                    }
                }
            }

            /*   for (int i = 0; i < Config.Configuration.NO_OCTAVES; i++)
               {
                   Console.WriteLine();
                   for (int j = 0; j < Config.Configuration.NO_BLUR_LVL; j++)
                   {
                       Console.Write(sigmas[i, j] + " ");
                   }

               }
               */
            ComputeLaplacianOfGaussian(octaves,sigmas);
        }
        public static void ComputeLaplacianOfGaussian(Octave[] octaves, double[,] sigmas)
        {
            //Aici intrebarea e: E bine ca am facut Math.Abs din diferenta?
            Octave[] LoG = new Octave[Config.Configuration.NO_OCTAVES];
            for (int i = 0; i < Config.Configuration.NO_OCTAVES; i++)
            {
                LoG[i] = new Octave();
                for (int j = 0; j < Config.Configuration.NO_BLUR_LVL-1; j++)
                {
                    
                    var rows = octaves[i].images[j].Rows;
                    var cols = octaves[i].images[j].Cols;
         
                    LoG[i].images[j] = new Mat(rows,cols,DepthType.Cv8U,1);
                    for (int ci = 0; ci < rows; ci++)
                    {
                        for (int cj = 0; cj < cols; cj++)
                        {
                            byte log1 = octaves[i].images[j].GetValue( ci, cj);
                            byte log2= octaves[i].images[j+1].GetValue( ci, cj);
                            LoG[i].images[j].SetValue( ci, cj, (byte)Math.Abs(log2 - log1));
                        }
                    }
                 //   CvInvoke.Imshow("grey", LoG[i].images[j]);
                 //   CvInvoke.WaitKey();
                }
            }
            FindLocalExtremes(LoG,sigmas);
       
        }

        public static void FindLocalExtremes(Octave[] octaves, double[,] sigmas)
        {
            //  Octave[] maxExtremes = new Octave[Config.Configuration.NO_OCTAVES];
            int no_extremes = 0;
            Octave[] extremes = new Octave[Config.Configuration.NO_OCTAVES];
            for (int i = 0; i < Config.Configuration.NO_OCTAVES - 1; i++)
            {   
                extremes[i] = new Octave();
            //    maxExtremes[i] = new Octave();
                for (int j = 1; j < Config.Configuration.NO_BLUR_LVL - 2; j++)
                {

                    var rows = octaves[i].images[j].Rows;
                    var cols = octaves[i].images[j].Cols;
                    extremes[i].images[j] =MatExtensions.InitMatWithZeros(rows, cols, DepthType.Cv8U, 1);
               //     maxExtremes[i].images[j] = MatExtensions.InitMatWithZeros(rows, cols, DepthType.Cv8U, 1);
                    for (int ci = 1; ci < rows-1; ci++)
                    {
                        for (int cj =1; cj < cols-1; cj++)
                        {
                            byte currentPoint = octaves[i].images[j].GetValue(ci, cj);
                            bool isMinExtreme = true;
                            bool isMaxExtreme = true;
                           
                            int lvl = j - 1;
                            while ((lvl < j + 2) && (isMinExtreme||isMaxExtreme))
                            {
                                int row = ci - 1;
                                while ((row < ci + 2) && (isMinExtreme || isMaxExtreme))
                                {
                                    int col = cj - 1;
                                    while ((col < cj + 2)&& (isMinExtreme || isMaxExtreme))
                                    {
                                        byte auxPoint = octaves[i].images[lvl].GetValue(row, col);
                                      
                                            if (currentPoint < auxPoint) isMaxExtreme = false;
                                       
                                            if (currentPoint > auxPoint) isMinExtreme = false;
                                      
                                        col++;
                                    }
                                    row++;
                                }

                                lvl++;
                            }
                            if (isMinExtreme || isMaxExtreme) {
                                extremes[i].images[j].SetValue(ci, cj, (byte)octaves[i].images[j].GetValue(ci, cj));
                                no_extremes++;
                            } 
                       
                        }
                    }
                }
            }
            /* for (int i = 0; i < Config.Configuration.NO_OCTAVES-1;i++)
             {
                 for (int j = 1; j < Config.Configuration.NO_BLUR_LVL - 2; j++)
                 {
                     if (localExtreme == "MAX")
                         CvInvoke.Imshow("grey", maxExtremes[i].images[j]);
                     else
                     CvInvoke.Imshow("grey", minExtremes[i].images[j]);
                     CvInvoke.WaitKey();
                 }
             }*/
            Console.WriteLine("Before corner detection: "+no_extremes);
            EliminateSomePointsWithComplicatedMathematics(extremes,sigmas);
        }

        private static void EliminateSomePointsWithComplicatedMathematics(Octave[] extremes, double[,] sigmas)
        {
            int numremoved = 0;
            //contrast check
            for (int i = 0; i < Config.Configuration.NO_OCTAVES - 1; i++)
            {
                for (int j = 1; j < Config.Configuration.NO_BLUR_LVL - 2; j++)
                {
                    var rows = extremes[i].images[j].Rows;
                    var cols = extremes[i].images[j].Cols;
                    for (int ci = 1; ci < rows - 1; ci++)
                    {
                        for (int cj = 1; cj < cols - 1; cj++)
                        {
                            if (extremes[i].images[j].GetValue(ci, cj) > 0)
                            {
                                if (Math.Abs(extremes[i].images[j].GetValue(ci,cj))<Config.Configuration.CONTRAST_THRESHOLD)
                                {
                                    numremoved++;
                                    extremes[i].images[j].SetValue(ci, cj, (byte)0);
                                }

                                double dii = extremes[i].images[j].GetValue(ci - 1, cj) +
                                    extremes[i].images[j].GetValue(ci + 1, cj) -
                                    2.0 * extremes[i].images[j].GetValue(ci, cj);

                                double djj = extremes[i].images[j].GetValue(ci , cj-1) +
                                    extremes[i].images[j].GetValue(ci, cj+1) -
                                    2.0 * extremes[i].images[j].GetValue(ci, cj);

                                double dij = extremes[i].images[j].GetValue(ci - 1, cj-1) +
                                    extremes[i].images[j].GetValue(ci+1, cj +1) +
                                   extremes[i].images[j].GetValue(ci-1 , cj + 1)+
                                   extremes[i].images[j].GetValue(ci +1, cj - 1) -
                                   2.0 * extremes[i].images[j].GetValue(ci, cj);

                                double trH = dii + djj;
                                double detH = dii * djj - dij * dij;

                                double curvature_ratio = trH * trH / detH;
                                if (detH<0 || curvature_ratio < Config.Configuration.CURVATURE_THRESHOLD)
                                {
                                    numremoved++;
                                    extremes[i].images[j].SetValue(ci, cj, (byte)0);
                                }
                            }

                          

                        }
                    }
                  
                }
            }
            Console.WriteLine("Removed" + numremoved);
            AssigningKeypointOrientation(extremes,sigmas);
        }
        public static void AssigningKeypointOrientation(Octave[] extremes, double[,] sigmas)
        {
            Octave[] magnitude = new Octave[Config.Configuration.NO_OCTAVES];
            Octave[] orientations = new Octave[Config.Configuration.NO_OCTAVES];
          
            for (int i = 0; i < Config.Configuration.NO_OCTAVES; i++)
            {
                int scale =(int) Math.Pow(2, i);
                for (int j = 1; j < extremes[i].images.Length+1; j++)
                {
                    Mat gx = new Mat();
                    Mat gy = new Mat();
                    CvInvoke.Sobel(extremes[i].images[j], gx, Emgu.CV.CvEnum.DepthType.Cv32F, 1, 0, 1);
                    CvInvoke.Sobel(extremes[i].images[j], gy, Emgu.CV.CvEnum.DepthType.Cv32F, 0, 1, 1);

                    double[,] mag = DataUtils.ComputeMagnitude(gx, gy);
                    double[,] angle =DataUtils.ComputeDirection(gx, gy);
                    double sigma = sigmas[i,j];
                    int window_size= GetKernelSize(1.5 * sigma);
                    int rows = extremes[i].images[j].Rows;
                    int cols= extremes[i].images[j].Cols;
                    for (int row = window_size/2; row < rows-window_size/2; row++)
                    {
                        for (int col = window_size / 2; col < cols- window_size / 2; cols++)
                        {
                            if (extremes[i].images[j].GetValue(row, col) > 0)
                            {
                                double[] hist = new double[10];
                                for (int ci = -window_size / 2; ci < window_size / 2; ci++)
                                {
                                    for (int cj = -window_size / 2; cj < window_size / 2; cj++)
                                    {
                                        double grad = angle[row + ci, col + cj];
                                        hist[(int)grad/10]+= mag[row + ci, col + cj];

                                    }
                                }
                                for (int ci = 0; ci < 10; ci++)
                                {
                                    ;
                                }
                            }
                        }
                    }
                }
            }

        }

        private static int GetKernelSize(object p)
        {
            int i;
            return 13;
        }

        public class Octave
        {
            public Mat[] images;
            public Octave()
            {
                images = new Mat[Config.Configuration.NO_BLUR_LVL];
            }
        }
    }
}
