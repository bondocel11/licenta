using DataLoader;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataPreparation
{
    public class DataSet
    {
        public List<InputEntry> Images { get; set; }

        public DataSet(List<InputEntry> images)
        {
            this.Images = images;
        }
    }
}
