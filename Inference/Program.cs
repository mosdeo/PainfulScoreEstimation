using Emgu.CV;
using LKY;
using System;
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
            var painfulFaceScoreEstimator = new PainfulFaceScoreEstimator();

            //初始化攝影機與人臉偵測
            Emgu.CV.VideoCapture webCam = new VideoCapture();
            CascadeClassifier faceDetect = new CascadeClassifier("./haarcascade_frontalface_alt2.xml");
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

                        double NN_Estimation_PSPI = painfulFaceScoreEstimator.getInferencePSPIScore(faceImg);
                        DrawFaceBox(RawImg, faceRect[face], NN_Estimation_PSPI);
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


        // 在攝影機畫面中，給被偵測到的臉部加上框，並且用字串把疼痛指數大小量視覺化
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