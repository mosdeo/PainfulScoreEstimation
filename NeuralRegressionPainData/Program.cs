using System;
using System.IO;
using System.Threading.Tasks;
using LKY;

namespace NeuralRegressionPainData
{
    class NeuralRegressionProgram
    {
        static int numInput = 1024; // usually more
        static int numHidden;
        static int numOutput = 1; // usual for regression
        static int rndSeed = 0;

        static int maxEpochs;
        static double learnRate;
        static double momentum;
        static double decay;
        static bool standardize = true;

        static String textStream = "";

        static private int GetFileRows(string fileName)
        {
            int num = 0;
            StreamReader sr = File.OpenText(fileName);
            while (sr.ReadLine() != null)
            { num += 1; }
            sr.Close();
            return num;
        }

        static double[][] ReadDataSet(FileInfo dataSet, bool standardize)
        {
            string Line;
            StreamReader SR = new StreamReader(dataSet.FullName);
            int numItems = GetFileRows(dataSet.FullName) - 1;//取得該DataSet總資料筆數(減掉第一行資料標籤)
            double[][] trainData = new double[numItems][];

            if (standardize)
            {
                string[][] RawTrainData = new string[trainData.Length][];

                for (int i = -1; (Line = SR.ReadLine()) != null; i++)
                {
                    if (-1 == i) continue;//跳過最初行
                    string[] ReadLine_Array = Line.Split(',');
                    string _PSPI = ReadLine_Array[1];
                    string[] painFeatureVector_PSPI = new string[numInput + 1];

                    for (int j = 0; j < numInput; j++)
                    {
                        painFeatureVector_PSPI[j] = ReadLine_Array[j + 2];
                    }

                    painFeatureVector_PSPI[numInput] = _PSPI;
                    RawTrainData[i] = painFeatureVector_PSPI;
                }

                string[] colTypes = new string[numInput + numOutput];
                for (int i = 0; i < colTypes.Length; i++) colTypes[i] = "numeric";
                Standardizer stder = new Standardizer(RawTrainData, colTypes);
                trainData = stder.StandardizeAll(RawTrainData);
            }
            else
            {//不標準化
                for (int i = -1; (Line = SR.ReadLine()) != null; i++)
                {
                    if (-1 == i) continue;//跳過最初行
                    string[] ReadLine_Array = Line.Split(',');
                    double _PSPI = Convert.ToDouble(ReadLine_Array[1]);
                    double[] painFeatureVector_PSPI = new double[numInput + 1];

                    for (int j = 0; j < numInput; j++)
                    {
                        painFeatureVector_PSPI[j] = Convert.ToDouble(ReadLine_Array[j + 2]);
                    }

                    painFeatureVector_PSPI[numInput] = _PSPI;
                    trainData[i] = painFeatureVector_PSPI;
                }
            }
            return trainData;            
        }

        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network Pain data regression demo\n");
            Console.WriteLine("Goal is to predict the PSPI\n");

            Console.Write("Hidden node:"); numHidden = Convert.ToInt32(Console.ReadLine());
            Console.Write("Iterations:"); maxEpochs = Convert.ToInt32(Console.ReadLine());
            Console.Write("Learn Rate:"); learnRate = Convert.ToDouble(Console.ReadLine());
            momentum = learnRate / 3;
            decay = learnRate / 100;

            // artificial; in realistic scenarios you'd read from a text file
            FileInfo trainingSet = new FileInfo(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFaceMaxPoolingPixel_32x32_DataAugmentation-SetA.csv");
            Console.WriteLine("\nProgrammatically reading " + GetFileRows(trainingSet.FullName) + " training data items");
            double[][] trainData = ReadDataSet(trainingSet, standardize);

            //呈現視覺化資料
            //視覺化.ShowPlot(trainData);

            Console.WriteLine("\nCreating a " + numInput + "-" + numHidden + "-" + numOutput + " regression neural network");
            Console.WriteLine("Using tanh hidden layer activation");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput, rndSeed);

            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F4"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F4"));

            Console.WriteLine("\nStarting training (using stochastic back-propagation)");
            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum);
            Console.WriteLine("===================  Finished training  ===================");

            //讀取所有驗證資料, 並標準化
            FileInfo testSet = new FileInfo(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFaceMaxPoolingPixel_32x32_DataAugmentation-SetB.csv");
            //FileInfo testSet = new FileInfo(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFace_32x32PoolingPixelValue-SetB.csv");
            Console.WriteLine("\nProgrammatically reading " + GetFileRows(testSet.FullName) + " test data items");
            double[][] testData = ReadDataSet(testSet, standardize);

            //驗證0~15範例, 並求出 MSE      
            for (int target=15,i=0;i<testData.Length && -1!=target; i++)
            {
                if (target == testData[i][testData[i].Length-1]) //inputVector[i][last] = PSPI
                {
                    double Predicted = nn.ComputeOutputs(testData[i])[0];
                    String str = String.Format("Actual PSPI = {0}   Predicted = {1}\n", target, Predicted);
                    Console.Write(str);
                    textStream += str;
                    target--;
                }
            }

            //求SetB 皮爾森積差
            double CORR = 0;
            double ActualPSPIAvg = 0, PredictedPSPIAvg = 0;
            double COVxy = 0, Sx = 0, Sy = 0;

            for (int i = 0; i < testData.Length; i++)
            {//算出實際和預測的平均值
                PredictedPSPIAvg += nn.ComputeOutputs(testData[i])[0];
                ActualPSPIAvg += testData[i][testData[i].Length - 1];
            }
            PredictedPSPIAvg /= testData.Length;
            ActualPSPIAvg /= testData.Length;

            for (int i = 0; i < testData.Length; i++)
            {//求差
                double Xerr = 0, Yerr = 0;
                Xerr = testData[i][testData[i].Length - 1] - ActualPSPIAvg;
                Yerr = nn.ComputeOutputs(testData[i])[0] - PredictedPSPIAvg;
                COVxy += Xerr * Yerr;
                Sx += Math.Pow(Xerr, 2);
                Sy += Math.Pow(Yerr, 2);
            }
            CORR = COVxy / Math.Pow(Sx * Sy, 0.5);
            String strCORR = String.Format("CORR = {0}\n", CORR);
            Console.Write(strCORR);
            textStream += strCORR;
            //結束

            //求SetB MSE
            double MSE = 0;
            for (int i = 0; i < testData.Length; i++)
            {
                //nn.ComputeOutputs只看建構子的numInput讀資料長度，所以testData[i]最後一項y-data會自動被忽略。
                double Predicted = nn.ComputeOutputs(testData[i])[0];
                double Actual = testData[i][testData[i].Length - 1];
                MSE += Math.Pow(Actual - Predicted, 2);
            }
            MSE = MSE / testData.Length;
            String strMSE = String.Format("MSE = {0}\n", MSE);
            Console.Write(strMSE);
            textStream += strMSE;

            //存檔
            textStream += String.Format("\n" + numInput + "-" + numHidden + "-" + numOutput + " regression neural network\n");
            textStream += String.Format("maxEpochs = " + maxEpochs+"\n");
            textStream += String.Format("learnRate = " + learnRate.ToString("F4") + "\n");
            textStream += String.Format("momentum  = " + momentum.ToString("F4") + "\n");

            textStream += String.Format("\nFinal neural network model weights:\n");
            double[] weightsOfNN = nn.GetWeights();
            foreach (double weight in weightsOfNN)
                textStream += String.Format("{0},",weight);

            textStream += String.Format("\n-----------------------------------------------------------------------\n");

            File.AppendAllText("trained.txt", textStream);
            Console.WriteLine("\nEnd demo\n");
            Console.ReadLine();
        } // Main
    }  // Program
} // ns
