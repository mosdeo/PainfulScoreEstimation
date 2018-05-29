using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using System.IO;
using LKY;

namespace Inference
{
    class Program
    {
        static void Main(string[] args)
        {
            //畫平均臉
            Mat ImgAvg = new Mat(32, 32, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            //Mat ImgAvg = new Mat("temp.jpg", Emgu.CV.CvEnum.LoadImageType.Color);
            //ImgAvg.SetTo(new Emgu.CV.Structure.MCvScalar(0xFF, 0xFF, 0xFF));
            for (int i = 0; i < ImgAvg.Rows; i++)
            {
                for (int j = 0; j < ImgAvg.Cols; j++)
                {
                    double sumPixel = 0;
                    for (int k = 0; k < 32; k++)
                    {
                        int index = (i * ImgAvg.Rows + j) * 32 + k;
                        sumPixel += 50*Math.Abs(WeightVector.PainFace_1024_32_1[index]);
                    }
                    sumPixel += 100 * (WeightVector.PainFace_1024_32_1[1024*32+i]);//add bias
                    MatExtension.SetValue(ImgAvg, i, j, (byte)sumPixel);
                }
            }
            CvInvoke.Resize(ImgAvg, ImgAvg, new System.Drawing.Size(256, 256), 0, 0, Emgu.CV.CvEnum.Inter.Nearest);
            CvInvoke.Imshow(nameof(ImgAvg), ImgAvg);
            CvInvoke.WaitKey();

            //畫平均臉
            //Mat ImgAvg = new Mat(32, 32, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            ////Mat ImgAvg = new Mat("temp.jpg", Emgu.CV.CvEnum.LoadImageType.Color);
            ////ImgAvg.SetTo(new Emgu.CV.Structure.MCvScalar(0xFF, 0xFF, 0xFF));
            //for (int i = 0; i < ImgAvg.Rows; i++)
            //{
            //    for (int j = 0; j < ImgAvg.Cols; j++)
            //    {
            //        MatExtension.SetValue(ImgAvg, i, j, Convert.ToByte(50+WeightVector.PainFace_1024_StdErr[i * ImgAvg.Rows + j]));
            //        //MatExtension.SetValue(ImgAvg, j, i, Convert.ToByte(0xFF));
            //    }
            //}
            //CvInvoke.Resize(ImgAvg, ImgAvg, new System.Drawing.Size(256, 256), 0, 0, Emgu.CV.CvEnum.Inter.Nearest);
            //CvInvoke.Imshow(nameof(ImgAvg), ImgAvg);
            //CvInvoke.WaitKey();
        }
    }
}
