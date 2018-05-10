using Config;
using DataLoader;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPreparation
{
    public class HogSets
    {
        public HogSet Train { get; set; }
        public HogSet Validation { get; set; }
        public HogSet Test { get; set; }
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

            var train_list = lis.GroupBy(x => x.Label).SelectMany(x => x.Take((int)(0.75 * x.Count()))).ToList();
            this.Train = new HogSet();
            this.Train.Load(train_list);


            this.Validation = new HogSet();
            this.Validation.Load(lis.Where(x => !train_list.Contains(x)).ToList());
            //  this.Test = new DataSet(testing_images);
           // this.Labels = reader.labels.Distinct().ToList();
            return true;
        }
    }
}
