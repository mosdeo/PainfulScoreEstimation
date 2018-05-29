using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;

namespace LKY
{//test
    class LBP
    {
        public static double[] GetUniformLBPfeature(Mat img)
        {
            Mat LBPimg = new Mat(img.Rows, img.Cols, Emgu.CV.CvEnum.DepthType.Cv8U, 1);  //複製一個同大小的
            return GetUniformLBPfeature(img, ref LBPimg);
        }

        public static double[] GetUniformLBPfeature(Mat img, ref Mat LBPimg)
        {
            int LBPsize = 1;
            double[] LBPhistogram = new double[59];
            Mat _LBPimg = new Mat(LBPimg.Rows, LBPimg.Cols, Emgu.CV.CvEnum.DepthType.Cv8U, 1);  //複製一個同大小的
            _LBPimg.SetTo(new Emgu.CV.Structure.MCvScalar(0xFF, 0xFF, 0xFF));//清除，不然會有雜點
            LBPimg.SetTo(new Emgu.CV.Structure.MCvScalar(0xFF, 0xFF, 0xFF));//清除，不然會有雜點

            for (int i = LBPsize; i < img.Rows - LBPsize; i += LBPsize)
            {
                for (int j = LBPsize; j < img.Cols - LBPsize; j += LBPsize)
                {
                    int LBPnumber = 0;
                    int centerBlockValue = MatExtension.GetValue(img, i, j);

                    LBPnumber += MatExtension.GetValue(img, i - 1, j - 1) < centerBlockValue ? 128 : 0;
                    LBPnumber += MatExtension.GetValue(img, i - 0, j - 1) < centerBlockValue ? 64 : 0;
                    LBPnumber += MatExtension.GetValue(img, i + 1, j - 1) < centerBlockValue ? 32 : 0;
                    LBPnumber += MatExtension.GetValue(img, i + 1, j - 0) < centerBlockValue ? 16 : 0;
                    LBPnumber += MatExtension.GetValue(img, i + 1, j + 1) < centerBlockValue ? 8 : 0;
                    LBPnumber += MatExtension.GetValue(img, i - 0, j + 1) < centerBlockValue ? 4 : 0;
                    LBPnumber += MatExtension.GetValue(img, i - 1, j + 1) < centerBlockValue ? 2 : 0;
                    LBPnumber += MatExtension.GetValue(img, i - 1, j - 0) < centerBlockValue ? 1 : 0;

                    LBPnumber = LBP.uniformLookupTable(LBPnumber);

                    //寫入結果
                    MatExtension.SetValue(_LBPimg, i, j, (byte)LBPnumber);
                    LBPhistogram[LBPnumber]++;
                }
            }

            LBPimg = _LBPimg.Clone(); //local 給 ref
            return LBPhistogram;
        }

        public static int uniformLookupTable(int LBPvalue)
        {
            int[] lookup = new int[256] { 0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };
            return lookup[LBPvalue];
        }

        public static double[] HistorgramNormalization(double[] srcArray, double max)
        {
            double sum = 0;
            foreach (double i in srcArray)
            {
                sum += i;
            }

            for (int i = 0; i < srcArray.Length; i++)
            {
                srcArray[i] = max * srcArray[i] / sum;
            }

            return srcArray;
        }

        public static Mat GetHistorgram(double[] arr)
        {
            int historgramHeight = 400;
            Mat historgram = new Mat(historgramHeight, 61, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            historgram.SetTo(new Emgu.CV.Structure.MCvScalar(0xFF, 0xFF, 0xFF));//清除，不然會有雜點

            for (int i = 0; i < 59; i++)
            {//由小到大依序產生
                for (int j = 1; j < Math.Min(historgramHeight, arr[i])/*防止繪圖超出範圍*/; j++)
                {//填滿每一直條
                    MatExtension.SetValue(historgram, j, i + 1, (byte)0);
                }
            }

            CvInvoke.Resize(historgram, historgram, new System.Drawing.Size(320, 240), 0, 0, Emgu.CV.CvEnum.Inter.Nearest);
            CvInvoke.Flip(historgram, historgram, Emgu.CV.CvEnum.FlipType.Vertical);
            return historgram;
        }
    }
}
