using DataLoader;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPreparation
{
    public class HogSet
    {
        public double[][] data { get; set; }
        public int[] outputs { get; set; }

        
        public void Load(List<InputEntry> list)
        {
            data = new double[list.Count()][];
            outputs = new int[list.Count()];
            int i = 0;
            List<string> labels = list.Select(x => x.Label).ToList();
            foreach (var entry in list)
            {
                double[] hog1 = DataUtils.DataUtils.HOG(entry.Image);
                outputs[i] = DataUtils.LabelTranslation.TranslateLabel(labels, entry.Label);
                HOGDescriptor hogCV = new HOGDescriptor(new Size(128, 128), new Size(16, 16), new Size(16,16), new Size(8, 8),9,1,-1,0.2,false);
                float[] hog = hogCV.Compute(entry.Image, new Size(16,16), Size.Empty, null);
                data[i] = hog.Select(x=>(double)x).ToArray();
                i++;
            }
        }
    }
}
