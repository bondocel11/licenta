using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataUtils
{
    public class HOGcell
    {
        public double[] Histogram;

        public HOGcell(double[] hist_mat)
        {
            this.Histogram = hist_mat;
        }

       
    }
}
