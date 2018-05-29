using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using System.IO;
using LKY;
using System.Diagnostics;

namespace RunTest
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
                System.Drawing.Rectangle[] faceRect = faceDetect.DetectMultiScale(RawImg, 1.1, 3, new System.Drawing.Size(128, 128));
                testTick = swTest.ElapsedMilliseconds;
                //如果沒偵測到人臉
                if (0 == faceRect.Length)
                {
                    CvInvoke.Imshow(nameof(RawImg), RawImg);
                }
                else
                {
                    //取出待測的臉部ROI
                    Mat faceImg = new Mat(RawImg, faceRect[0]);
                    CvInvoke.Resize(faceImg, faceImg, new System.Drawing.Size(64, 64),0,0);
                    CvInvoke.CvtColor(faceImg, faceImg, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                    //原始灰階+直方圖等化
                    CvInvoke.Imshow("Bgr2Gray", faceImg);
                    CvInvoke.EqualizeHist(faceImg, faceImg);

                    ////Flip
                    //Mat flipFace = new Mat();
                    //CvInvoke.Flip(faceImg, flipFace, Emgu.CV.CvEnum.FlipType.Horizontal);
                    //CvInvoke.Imshow("FlipH", flipFace);

                    ////Equalize
                    //Mat equFace = new Mat();
                    //CvInvoke.EqualizeHist(faceImg, equFace);
                    //CvInvoke.Imshow("HistEqu", equFace);

                    //CvInvoke.EqualizeHist(faceImg, faceImg);
                    //CvInvoke.Flip(faceImg, faceImg, Emgu.CV.CvEnum.FlipType.Horizontal);
                    //CvInvoke.Imshow("HistEqu+FlipH", faceImg);

                    //Max Pooling
                    double[] inputVector;
                    
                    Mat PoolingImg = Pooling.Do(faceImg, true, out inputVector);
                    //使用預設的平均值和標準差,做高斯正規化
                    
                    for (int i = 0; i < 1024; i++)
                    //Parallel.For(0, 1024, i =>
                    {
                        inputVector[i] =(inputVector[i] - WeightVector.PainFace_1024_DA_Avg[i]) / WeightVector.PainFace_1024_DA_StdErr[i];
                    }
                    //);
                    
                    double NN_Estimation_PSPI = nn.ComputeOutputs(inputVector)[0];   
                    Console.Clear();
                    CvInvoke.Imshow(nameof(RawImg), RawImg);
                    CvInvoke.Imshow(nameof(faceImg), faceImg);
                    int scalePoolImg = 6;
                    CvInvoke.Resize(PoolingImg, PoolingImg, new System.Drawing.Size(scalePoolImg*PoolingImg.Cols, scalePoolImg * PoolingImg.Rows), 0, 0, Emgu.CV.CvEnum.Inter.Nearest);
                    CvInvoke.Imshow(nameof(PoolingImg), PoolingImg);
                    Console.WriteLine(nameof(testTick)+"={0:0.00}", testTick);
                    Console.WriteLine("FPS={0:0.00}", 1000/aLoopTime);
                    Console.WriteLine("{0} = {1:0.00}", nameof(NN_Estimation_PSPI), NN_Estimation_PSPI);
                    Console.Write("Pain = "); for (int i = 0; i < NN_Estimation_PSPI; i++) { Console.Write("▉"); }

                    //存一些範例
                    //CvInvoke.Imwrite(nameof(faceImg)+".png", faceImg);
                    //CvInvoke.Imwrite(nameof(PoolingImg)+".png", PoolingImg);
                }
                CvInvoke.WaitKey(1);
            }
        }
    }
}
