using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using System.IO;
using LKY;

namespace RunTest
{
    class Program
    {
        static void Main(string[] args)
        {
            double AllMSE = 0;
            int frameCount = 0;

            //初始化類神經網路
            int numInput = 1024; // usually more
            int numHidden = 32;
            int numOutput = 1; // usual for regression
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput, 0);
            double[] weights = WeightVector.PainFace_1024_32_1_No3;
            nn.SetWeights(weights);

            //初始化人臉偵測
            CascadeClassifier faceDetect = new CascadeClassifier("../../haarcascade_frontalface_alt2.xml");

            //抓取受試者檔案路徑清單
            DirectoryInfo[] subjectList = new DirectoryInfo(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\UNBC-McMaster Shoulder Pain\Images").GetDirectories();

            //掃描每個受試者
            foreach (DirectoryInfo subject in subjectList)
            {//25 times

                DirectoryInfo[] VideoList = subject.GetDirectories();

                //掃描每個影片
                foreach (DirectoryInfo _dir in VideoList)
                {
                    //存BenchMark圖所需資料
                    String[] csvBenchMarkDiagram = new String[2];
                    csvBenchMarkDiagram[0] = "Predicted = [";
                    csvBenchMarkDiagram[1] = "Actual = [";

                    //計算用
                    double MSE = 0;
                    int maxPain = 0;

                    //單一影片的所有frame
                    FileInfo[] picList = _dir.GetFiles("*.png", SearchOption.AllDirectories);

                    //掃描每個frame
                    foreach (FileInfo pic in picList)
                    {
                        frameCount++;

                        //讀取PSPI
                        string strGroundTruthPath = pic.FullName;
                        strGroundTruthPath = strGroundTruthPath.Replace("\\Images", "\\Frame_Labels\\PSPI");
                        strGroundTruthPath = strGroundTruthPath.Replace(".png", "_facs.txt");
                        double Actual = Convert.ToDouble(new System.IO.StreamReader(strGroundTruthPath).ReadLine());
                        double Predicted = 0;

                        maxPain = Math.Max(maxPain, Convert.ToInt32(Actual));

                        //讀檔 //偵測人臉, 限定尺寸下限, 加快速度&過濾錯誤
                        Mat RawImg = CvInvoke.Imread(pic.FullName, Emgu.CV.CvEnum.LoadImageType.AnyColor);
                        System.Drawing.Rectangle[] faceRect = faceDetect.DetectMultiScale(RawImg, 1.1, 3, new System.Drawing.Size(70, 70));

                        //如果偵測到人臉
                        if (0 != faceRect.Length)
                        {
                            //取出待測的臉部ROI, 並顯示
                            Mat faceRegion = new Mat(RawImg, faceRect[0]);
                            CvInvoke.Resize(faceRegion, faceRegion, new System.Drawing.Size(64, 64));
                            CvInvoke.CvtColor(faceRegion, faceRegion, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
                            CvInvoke.EqualizeHist(faceRegion, faceRegion);

                            //Max Pooling
                            double[] inputVector;
                            Pooling.Do(faceRegion, true, out inputVector);

                            for (int i = 0; i < 1024; i++)
                            {//使用預設的平均值和標準差,做高斯正規化
                                inputVector[i] = (inputVector[i] - WeightVector.PainFace_1024_Avg[i]) / WeightVector.PainFace_1024_StdErr[i];
                            }

                            Predicted = nn.ComputeOutputs(inputVector)[0];
                        }

                        Predicted = Math.Max(0, Predicted); //range limit
                        csvBenchMarkDiagram[0] += String.Format("{0} ", Predicted);
                        csvBenchMarkDiagram[1] += String.Format("{0} ", Actual);
                        MSE += Math.Pow(Actual - Predicted, 2);
                        AllMSE += Math.Pow(Actual - Predicted, 2);

                        Console.Clear();
                        Console.Write(pic.Name);
                    }
                    Console.WriteLine("影片{0}掃描完畢", _dir.Name);

                    if (3 < maxPain)
                    {
                        csvBenchMarkDiagram[0] += "]'; % Known groups\r\n";
                        csvBenchMarkDiagram[1] += "]'; % Predicted groups\r\n";
                        csvBenchMarkDiagram[1] += "plot([1:1:length(Predicted)],Predicted,'g',[1:1:length(Actual)],Actual,'r--');\r\n";
                        csvBenchMarkDiagram[1] += "h=legend('Estimation', 'Ground Truth'); set(h,'FontSize', 14)\r\n";
                        csvBenchMarkDiagram[1] += String.Format("title('{0}','FontSize', 14); xlabel('{1}', 'FontSize', 14); ylabel('{2}', 'FontSize', 14);\r\n", picList.First().Name, "Frame number", "PSPI score");
                        csvBenchMarkDiagram[1] += "xlim([0 max(length(Predicted),length(Actual))]);\r\n";
                        csvBenchMarkDiagram[1] += "ylim([-0.5 max(max(Predicted),max(Actual))+1]);\r\n";
                        csvBenchMarkDiagram[1] += String.Format("% {0}, MSE = {1:0.000}, Max Pain={2}, {3}\r\n", picList.First().Name, MSE / picList.Length, maxPain, DateTime.Now);
                        csvBenchMarkDiagram[1] += "--------------------------------------------------------------------------\r\n";                
                        File.AppendAllText("UNBC_Pain_SequenceBenchmark.txt", csvBenchMarkDiagram[0] + csvBenchMarkDiagram[1]);
                    }
                    Console.WriteLine(csvBenchMarkDiagram[0] + csvBenchMarkDiagram[1]);
                }
            }

            File.AppendAllText("UNBC_Pain_SequenceBenchmark.txt", String.Format("All MSE = {0}", AllMSE / frameCount));
            Console.WriteLine("\nEnd");
            Console.ReadLine();
        }
    }
}