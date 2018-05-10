using DataUtils;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPreparation
{
    public static class SIFT
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
            ComputeLaplacianOfGaussian(octaves);
        }
        public static void ComputeLaplacianOfGaussian(Octave[] octaves)
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
            FindLocalExtremes(LoG,"MAX");
            FindLocalExtremes(LoG, "MIN");
        }

        public static void FindLocalExtremes(Octave[] octaves,string localExtreme)
        {
            Octave[] maxExtremes = new Octave[Config.Configuration.NO_OCTAVES];
            Octave[] minExtremes = new Octave[Config.Configuration.NO_OCTAVES];
            for (int i = 0; i < Config.Configuration.NO_OCTAVES - 1; i++)
            {
                minExtremes[i] = new Octave();
                maxExtremes[i] = new Octave();
                for (int j = 1; j < Config.Configuration.NO_BLUR_LVL - 2; j++)
                {

                    var rows = octaves[i].images[j].Rows;
                    var cols = octaves[i].images[j].Cols;
                    minExtremes[i].images[j] =MatExtensions.InitMatWithZeros(rows, cols, DepthType.Cv8U, 1);
                    maxExtremes[i].images[j] = MatExtensions.InitMatWithZeros(rows, cols, DepthType.Cv8U, 1);
                    for (int ci = 1; ci < rows-1; ci++)
                    {
                        for (int cj =1; cj < cols-1; cj++)
                        {
                            byte currentPoint = octaves[i].images[j].GetValue(ci, cj);
                            bool isExtreme = true;
                            int lvl = j - 1;
                            while ((lvl < j + 2) && isExtreme)
                            {
                                int row = ci - 1;
                                while ((row < ci + 2) && isExtreme)
                                {
                                    int col = cj - 1;
                                    while ((col < cj + 2)&& isExtreme)
                                    {
                                        byte auxPoint = octaves[i].images[lvl].GetValue(row, col);
                                        if (localExtreme == "MAX")
                                        {
                                            
                                            if (currentPoint < auxPoint) isExtreme = false;
                                        }
                                        else
                                        {
                                  
                                            if (currentPoint > auxPoint) isExtreme = false;
                                        }
                                        col++;
                                    }
                                    row++;
                                }

                                lvl++;
                            }
                            if (localExtreme=="MAX" && isExtreme) maxExtremes[i].images[j].SetValue(ci, cj, (byte)octaves[i].images[j].GetValue(ci, cj));
                            if (localExtreme == "MIN" && isExtreme) minExtremes[i].images[j].SetValue(ci, cj,(byte)octaves[i].images[j].GetValue(ci, cj));
                       
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

            EliminateSomePointsWithComplicatedMathematics(maxExtremes, minExtremes);
        }

        private static void EliminateSomePointsWithComplicatedMathematics(Octave[] maxExtremes, Octave[] minExtremes)
        {
            int numremoved = 0;
            //contrast check
            for (int i = 0; i < Config.Configuration.NO_OCTAVES - 1; i++)
            {
                for (int j = 1; j < Config.Configuration.NO_BLUR_LVL - 2; j++)
                {
                    var rows = maxExtremes[i].images[j].Rows;
                    var cols = maxExtremes[i].images[j].Cols;
                    for (int ci = 1; ci < rows - 1; ci++)
                    {
                        for (int cj = 1; cj < cols - 1; cj++)
                        {
                            if (minExtremes[i].images[j].GetValue(ci, cj) > 0)
                            {
                                if (Math.Abs(CvInvoke.cvGetReal2D(maxExtremes[i].images[j], ci, cj))<Config.Configuration.CONTRAST_THRESHOLD)
                                {
                                    numremoved++;
                                    minExtremes[i].images[j].SetValue(ci, cj, 0);
                                }

                                
                            }
                                


                        }
                    }
                }
            }
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
