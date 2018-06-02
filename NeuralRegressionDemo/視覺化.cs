using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using LKY;

namespace LKY
{
    class 視覺化
    {
        public static void ShowPlot(double[][] trainData)
        {
            //繪製視覺化
            int RectSize = 56*2;
            Mat img = new Mat(RectSize, RectSize, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            img.SetTo(new Emgu.CV.Structure.MCvScalar(0x0, 0x0, 0x0));//清除，不然會有雜點

            for (int i = 0; i < trainData.Length; i++)
            {
                MatExtension.SetValue(img, (int)(RectSize/4 * trainData[i][1]) + RectSize/2, (int)(trainData[i][0]*RectSize / 7), (byte)0xFF);
            }

            CvInvoke.Flip(img, img, Emgu.CV.CvEnum.FlipType.Vertical);
            CvInvoke.Resize(img, img, new System.Drawing.Size(56*4, 56 * 4), 0, 0, Emgu.CV.CvEnum.Inter.Nearest);
            CvInvoke.Imshow("img", img);
            CvInvoke.WaitKey(1);
        }

        //public static void ShowPlot(double[] trainData)
        //{
        //    //繪製視覺化
        //    Mat img = new Mat(160, 160, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
        //    img.SetTo(new Emgu.CV.Structure.MCvScalar(0xFF, 0xFF, 0xFF));//清除，不然會有雜點

        //    for (int i = 0; i < trainData.Length; i++)
        //    {
        //        MatExtension.SetValue(img, (int)(10 * trainData[i]) + 100, i*20, (byte)0);
        //    }
        //    CvInvoke.Flip(img, img, Emgu.CV.CvEnum.FlipType.Vertical);
        //    CvInvoke.Resize(img, img, new System.Drawing.Size(320, 320), 0, 0, Emgu.CV.CvEnum.Inter.Nearest);
        //    CvInvoke.Imshow("img", img);
        //    CvInvoke.WaitKey(1);
        //}
    }
}
