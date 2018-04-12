using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataLoader { 
    public class InputEntry
    {
        public Mat Image { get; set; }

        public string Label { get; set; }

        public override string ToString()
        {
            return "Label: " + this.Label;
        }
    }
}
