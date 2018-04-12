using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataLoader
{
    public class DataReader
    {
        public List<Mat> images { get; set; }
        public List<string> labels { get; set; }


        public void LoadData(string filename)
        {
            
            if (Directory.Exists(filename))
            {
                images = new List<Mat>();
                labels = new List<string>();
                string[] subdirEntries = Directory.GetDirectories(filename);
                foreach (string dirName in subdirEntries)
                {
                    string[] fileEntries = Directory.GetFiles(dirName);
                    

                    for (int i = 0; i < fileEntries.Length; i++)
                    {
                        if (fileEntries[i].EndsWith(".jpg", StringComparison.Ordinal))
                        {
                            Mat modelImage = CvInvoke.Imread(fileEntries[i], ImreadModes.Color);
                            images.Add(modelImage);
                            labels.Add(dirName.Remove(0, filename.Length+1).ToLower());
                        }
                            
                    }
                }
            }
        }
    }
}
