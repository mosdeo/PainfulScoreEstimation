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
            //初始化類神經網路
            int numInput = 1024; // usually more
            int numHidden = 32;
            int numOutput = 1; // usual for regression
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput, 0);
            double[] weights = WeightVector.PainFace_1024_32_1_No2;
            nn.SetWeights(weights);

            //讀取所有驗證資料, 並標準化
            int numItems = 359;
            double[][] verificationData = new double[numItems][];

            Console.WriteLine("\nProgrammatically reading " + numItems + " Verification data items");

            double[][] inputVector = new double[numItems][];
            using (StreamReader SR = new StreamReader(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFace_32x32PoolingPixelValue-jh123t1aeaff.csv"))
            {
                string Line;
                string[][] RawVerificationData = new string[verificationData.Length][];
                //while ((Line = SR.ReadLine()) != null)
                for (int i = -1; (Line = SR.ReadLine()) != null; i++)
                {
                    if (-1 == i) continue;//跳過最初行(不讀取檔案名稱)

                    string[] ReadLine_Array = Line.Split(',');
                    string _PSPI = ReadLine_Array[1];
                    string[] uniLBPfeature118Vector_PSPI = new string[numInput + 1];

                    for (int j = 0; j < numInput; j++)
                    {
                        uniLBPfeature118Vector_PSPI[j] = ReadLine_Array[j + 2];
                    }

                    uniLBPfeature118Vector_PSPI[numInput] = _PSPI;
                    RawVerificationData[i] = uniLBPfeature118Vector_PSPI;
                }

                string[] colTypes = new string[numInput + numOutput];
                for (int i = 0; i < colTypes.Length; i++) colTypes[i] = "numeric";
                Standardizer stder = new Standardizer(RawVerificationData, colTypes);
                inputVector = stder.StandardizeAll(RawVerificationData);
            }

            //存散布圖所需資料
            String[] csvBenchMarkDiagram = new String[2];
            csvBenchMarkDiagram[0] = "Predicted = [";
            csvBenchMarkDiagram[1] = "Actual = [";

            for (int i = 0; i < inputVector.Length; i++)
            {//算出實際和預測
                double Predicted = nn.ComputeOutputs(inputVector[i])[0];
                double Actual = inputVector[i][inputVector[i].Length - 1];
                Predicted = Math.Min(15, Math.Max(0, Predicted)); //range limit
                csvBenchMarkDiagram[0] += String.Format("{0} ", Math.Round(Predicted));
                csvBenchMarkDiagram[1] += String.Format("{0} ", Actual);
            }
            csvBenchMarkDiagram[0] += "]'; % Known groups\r\n";
            csvBenchMarkDiagram[1] += "]'; % Predicted groups\r\n";
            csvBenchMarkDiagram[1] += "plot([1:1:length(Predicted)],Predicted,'g',[1:1:length(Actual)],Actual,'r--');\r\n";
            csvBenchMarkDiagram[1] += "legend('Predicted', 'Actual');\r\n";
            csvBenchMarkDiagram[1] += "xlim([0 max(length(Predicted),length(Actual))]);\r\n";
            csvBenchMarkDiagram[1] += "ylim([-0.5 max(max(Predicted),max(Actual))+1]);";
            csvBenchMarkDiagram[1] += "--------------------------------------------------------------------------\r\n";

            File.AppendAllText("BenchMarkDiagram.txt", csvBenchMarkDiagram[0] + csvBenchMarkDiagram[1]);

            Console.WriteLine("\nEnd");
            Console.ReadLine();
        }
    }
}