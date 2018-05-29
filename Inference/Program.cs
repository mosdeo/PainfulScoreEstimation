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
            String textStream = "";
            int numItems = 23320;
            double[][] trainData = new double[numItems][];

            Console.WriteLine("\nProgrammatically reading " + numItems + " Verification data items");

            double[][] inputVector = new double[numItems][];
            using (StreamReader SR = new StreamReader(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFace_32x32PoolingPixelValue-SetB.csv"))
            {
                string Line;
                string[][] RawVerificationData = new string[trainData.Length][];
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

            //驗證0~15範例, 並求出 MSE      
            for (int target = 15, i = 0; i < inputVector.Length && -1 != target; i++)
            {
                if (target == inputVector[i][inputVector[i].Length - 1]) //inputVector[i][last] = PSPI
                {
                    double Predicted = nn.ComputeOutputs(inputVector[i])[0];
                    String str = String.Format("Actual PSPI = {0}   Predicted = {1}\n", target, Predicted);
                    Console.Write(str);
                    textStream += str;
                    target--;
                }
            }
            //求SetB 皮爾森積差
            double CORR = 0;
            double ActualPSPIAvg = 0, PredictedPSPIAvg = 0;
            double COVxy = 0, Sx=0, Sy = 0;

            for (int i = 0; i < inputVector.Length; i++)
            {//算出實際和預測的平均值
                double Predicted = nn.ComputeOutputs(inputVector[i])[0];
                double Actual = inputVector[i][inputVector[i].Length - 1];
                PredictedPSPIAvg += Predicted;
                ActualPSPIAvg += Actual;
            }
            PredictedPSPIAvg /= inputVector.Length;
            ActualPSPIAvg /= inputVector.Length;

            for (int i = 0; i < inputVector.Length; i++)
            {//求差
                double Xerr = 0, Yerr = 0;
                Xerr = inputVector[i][inputVector[i].Length - 1] - ActualPSPIAvg;
                Yerr = nn.ComputeOutputs(inputVector[i])[0] - PredictedPSPIAvg;
                COVxy += Xerr * Yerr;
                Sx += Math.Pow(Xerr, 2);
                Sy += Math.Pow(Yerr, 2);
            }
            CORR = COVxy/Math.Pow(Sx*Sy, 0.5);
            String strCORR = String.Format("CORR = {0}\n", CORR);
            Console.Write(strCORR);
            textStream += strCORR;
            //結束

            //求SetB MSE
            double MSE = 0;
            for (int i = 0; i < inputVector.Length; i++)
            {
                //nn.ComputeOutputs只看建構子的numInput讀資料長度，所以inputVector[i]最後一項y-data會自動被忽略。
                double Predicted = nn.ComputeOutputs(inputVector[i])[0];
                double Actual = inputVector[i][inputVector[i].Length - 1];
                MSE += Math.Pow(Actual - Predicted, 2);
            }
            MSE = MSE / inputVector.Length;
            String strMSE = String.Format("MSE = {0}\n", MSE);
            Console.Write(strMSE);
            textStream += strMSE;
            //結束


            //存散布圖所需資料
            //String[] csvScatterDiagram = new String[2];
            //csvScatterDiagram[0] = "Predicted = [";
            //csvScatterDiagram[1] = "Actual = [";

            //for (int i = 0; i < inputVector.Length; i++)
            //{//算出實際和預測的平均值
            //    double Predicted = nn.ComputeOutputs(inputVector[i])[0];
            //    double Actual = inputVector[i][inputVector[i].Length - 1];
            //    Predicted = Math.Min(15, Math.Max(0, Predicted)); //range limit
            //    csvScatterDiagram[0] += String.Format("{0} ", Math.Round(Predicted));
            //    csvScatterDiagram[1] += String.Format("{0} ", Actual);
            //}
            //csvScatterDiagram[0] += "]'; % Known groups\r\n";
            //csvScatterDiagram[1] += "]'; % Predicted groups\r\n";
            //csvScatterDiagram[1] += "[confusion_matrix, order] = confusionmat(Actual, Predicted);\r\n";
            //csvScatterDiagram[1] += "for i = 1:16\r\n";
            //csvScatterDiagram[1] += "   confusion_matrix([i],:) = confusion_matrix([i],:)/ sum(confusion_matrix([i],:));\r\n";
            //csvScatterDiagram[1] += "end\r\n";
            //csvScatterDiagram[1] += "imshow(confusion_matrix, 'InitialMagnification',2000);\r\n";
            //csvScatterDiagram[1] += "colormap(jet) % # to change the default grayscale colormap\r\n";
            //csvScatterDiagram[1] += "title('1024-32-1 Neural Network Regression');\r\n";
            //csvScatterDiagram[1] += "xlabel('類神經網路估測值');\r\n";
            //csvScatterDiagram[1] += "ylabel('Ground Truth');\r\n";
            //csvScatterDiagram[1] += "colorbar;\r\n";

            //File.AppendAllText("ScatterDiagram.txt", csvScatterDiagram[0]+ csvScatterDiagram[1]);


            Console.ReadLine();
        }

        //static void Main(string[] args)
        //{
        //    NeuralNetwork nn = new NeuralNetwork(1, 3, 1, 0);
        //    double[] weights = WeightVector.Sin_1_3_1;
        //    nn.SetWeights(weights);

        //    double[] y = nn.ComputeOutputs(new double[] { Math.PI });
        //    Console.WriteLine("\nActual sin(PI)       =  0.0   Predicted =  " + y[0].ToString("F6"));

        //    y = nn.ComputeOutputs(new double[] { Math.PI / 2 });
        //    Console.WriteLine("\nActual sin(PI / 2)   =  1.0   Predicted =  " + y[0].ToString("F6"));

        //    y = nn.ComputeOutputs(new double[] { 3 * Math.PI / 2.0 });
        //    Console.WriteLine("\nActual sin(3*PI / 2) = -1.0   Predicted = " + y[0].ToString("F6"));

        //    y = nn.ComputeOutputs(new double[] { 6 * Math.PI });
        //    Console.WriteLine("\nActual sin(6*PI)     =  0.0   Predicted =  " + y[0].ToString("F6"));

        //    Console.ReadLine();
        //}
    }
}
