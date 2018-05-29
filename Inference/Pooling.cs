using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Emgu.CV;

namespace LKY
{
    class Pooling
    {
        public static Mat Do(Mat srcImg, bool isMax)
        {
            double[] inputVector = new double[srcImg.Rows / 2 * srcImg.Cols / 2];
            return Do(srcImg, isMax, out inputVector);
        }

        public static Mat Do(Mat srcImg, bool isMax, out double[] inputVector)
        {
            //Max Pooling
            Mat poolingImg = new Mat(srcImg.Rows / 2, srcImg.Cols / 2, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            inputVector = new double[poolingImg.Rows * poolingImg.Cols];

            for (int j = 0; j < poolingImg.Rows; j++)
            {
                for (int i = 0; i < poolingImg.Cols; i++)
                {
                    int[] PixelPool = new int[4];
                    PixelPool[0] = MatExtension.GetValue(srcImg, 2 * j, 2 * i);
                    PixelPool[1] = MatExtension.GetValue(srcImg, 2 * j + 1, 2 * i);
                    PixelPool[2] = MatExtension.GetValue(srcImg, 2 * j, 2 * i + 1);
                    PixelPool[3] = MatExtension.GetValue(srcImg, 2 * j + 1, 2 * i + 1);
                    MatExtension.SetValue(poolingImg, j, i, isMax ? (byte)PixelPool.Max() : (byte)PixelPool.Min());
                    inputVector[j * poolingImg.Cols + i] = Convert.ToDouble(isMax ? (byte)PixelPool.Max() : (byte)PixelPool.Min());
                }
            }
            return poolingImg;
        }
    }
}
