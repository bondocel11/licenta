using Config;
using DataLoader;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPreparation
{
    public class DataSets
    {

        public DataSet Train { get; set; }

        public DataSet Validation { get; set; }

        public DataSet Test { get; set; }

        public List<string> Labels { get; set; }

        public bool Load(int validationSize = 100)
        {
            // Load data
            Console.WriteLine("Loading the datasets...");
            DataReader reader = new DataReader();
            reader.LoadData(Configuration.trainingImageFile);
            var raw_train_images = reader.images;
            //var testing_images = DataReader.Load(Configuration.testingLabelFile, Configuration.testingImageFile);
           
            var ready_train_images = DataMolding.Prepare(raw_train_images);

            List<InputEntry> lis = reader.labels.Select((t, i) => new InputEntry { Label = t, Image = ready_train_images[i] }).ToList();

            this.Train = new DataSet(lis.GetRange(ready_train_images.Count - validationSize, validationSize));
            this.Validation = new DataSet(lis.GetRange(0, ready_train_images.Count - validationSize));
            //  this.Test = new DataSet(testing_images);
            this.Labels = reader.labels;
            return true;
        }
    }
}
