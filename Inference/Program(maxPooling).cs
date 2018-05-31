using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using System.IO;
using LKY;
using System.Diagnostics;

namespace Inference
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.BackgroundColor = ConsoleColor.DarkCyan;
            Console.ForegroundColor = ConsoleColor.White;

            //初始化類神經網路
            NeuralNetwork nn = new NeuralNetwork(1024, 32, 1, 0);
            double[] weights = WeightVector.PainFace_1024_DA;
            nn.SetWeights(weights);

            //初始化攝影機與人臉偵測
            Emgu.CV.VideoCapture webCam = new VideoCapture();
            CascadeClassifier faceDetect = new CascadeClassifier("../../haarcascade_frontalface_alt2.xml");
            Stopwatch sw = new Stopwatch();
            Stopwatch swTest = new Stopwatch();
            double aLoopTime;
            double testTick;

            while (true)
            {
                aLoopTime = sw.ElapsedMilliseconds;
                sw.Restart();
                Mat RawImg = webCam.QueryFrame();
                CvInvoke.Flip(RawImg, RawImg, Emgu.CV.CvEnum.FlipType.Horizontal);

                //偵測人臉, 限定尺寸下限, 加快速度&過濾錯誤
                swTest.Restart();
                System.Drawing.Rectangle[] faceRect = faceDetect.DetectMultiScale(RawImg, 1.1, 10, new System.Drawing.Size(70, 70));
                testTick = swTest.ElapsedMilliseconds;
                //如果偵測到人臉
                if (0 != faceRect.Length)
                {
                    for (int face = 0; face < faceRect.Length; face++)
                    {
                        //取出待測的臉部ROI
                        Mat faceImg = new Mat(RawImg, faceRect[face]);
                        CvInvoke.Resize(faceImg, faceImg, new System.Drawing.Size(64, 64), 0, 0);
                        CvInvoke.CvtColor(faceImg, faceImg, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
                        CvInvoke.EqualizeHist(faceImg, faceImg);

                        //Max Pooling
                        double[] inputVector;

                        Mat PoolingImg = Pooling.Do(faceImg, true, out inputVector);
                        //使用預設的平均值和標準差,做高斯正規化

                        for (int i = 0; i < 1024; i++)
                        //Parallel.For(0, 1024, i =>
                        {
                            inputVector[i] = (inputVector[i] - WeightVector.PainFace_1024_DA_Avg[i]) / WeightVector.PainFace_1024_DA_StdErr[i];
                        }
                        //);

                        double NN_Estimation_PSPI = nn.ComputeOutputs(inputVector)[0];

                        DrawFaceBox(RawImg, faceRect[face], NN_Estimation_PSPI);

                        //if (NN_Estimation_PSPI > MaxPSPI)
                        //{
                        //    CvInvoke.Imwrite("PSPI=" + Convert.ToInt32(NN_Estimation_PSPI).ToString() + ".png", RawImg);
                        //    MaxPSPI = NN_Estimation_PSPI;
                        //}
                    }
                }

                //寫入畫面上的文字訊息
                String fps = String.Format("FPS={0:0.00}", 1000 / aLoopTime);
                CvInvoke.PutText(RawImg, fps, new System.Drawing.Point(0, 30), 0, 1, new Emgu.CV.Structure.MCvScalar(51, 150, 255), 1);

                CvInvoke.Imshow(nameof(RawImg), RawImg);
                //vw.Write(RawImg);
                CvInvoke.WaitKey(1);
            }
        }

        static void DrawFaceBox(Mat frameFaceBox, System.Drawing.Rectangle faceRect, double PSPI)
        {
            Emgu.CV.Structure.MCvScalar colorOrange = new Emgu.CV.Structure.MCvScalar(51, 150, 255);
            System.Drawing.Point textPosition = new System.Drawing.Point(faceRect.X, faceRect.Y + 22 + faceRect.Size.Height);

            String painNum = String.Format("PSPI={0:0.0}", PSPI);
            String painBar = "";
            for (int i = 0; i < PSPI; i++) { painBar += "*"; }
            //Console.Write("Pain = "); for (int i = 0; i < NN_Estimation_PSPI; i++) { Console.Write("▉"); }

            CvInvoke.Rectangle(frameFaceBox, faceRect, colorOrange, 1);
            CvInvoke.PutText(frameFaceBox, painNum, textPosition, Emgu.CV.CvEnum.FontFace.HersheyComplex, 1, colorOrange, 2);

            textPosition.Y = faceRect.Y;
            CvInvoke.PutText(frameFaceBox, painBar, textPosition, Emgu.CV.CvEnum.FontFace.HersheyComplex, 1, colorOrange, 2);
        }
    }
}