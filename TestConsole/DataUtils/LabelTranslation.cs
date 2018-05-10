using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataUtils
{
    public class LabelTranslation
    {
        public static int[] TranslateLabels(List<string> allLabels, string[] testLabels)
        {

            return testLabels.Select(label => allLabels.IndexOf(label)).ToArray();
        }

        public static int TranslateLabel(List<string> allLabels,string label)
        {
            return allLabels.Distinct().ToList().IndexOf(label);
        }
    }
}
