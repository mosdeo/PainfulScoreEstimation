using System;
using System.IO;
using System.Threading.Tasks;
using LKY;

namespace NeuralRegressionPainData
{
    class NeuralRegressionProgram
    {
        static int numInput = 4096; // usually more
        static int numHidden = 32;
        static int numOutput = 1; // usual for regression
        static int rndSeed = 0;

        static int maxEpochs = 1000;
        static double learnRate = 0.005;
        static double momentum = 0.001;

        static String textStream = "";

        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network Pain data regression demo\n");
            Console.WriteLine("Goal is to predict the PSPI\n");

            Console.Write("Hidden node:"); numHidden = Convert.ToInt32(Console.ReadLine());
            Console.Write("Iterations:"); maxEpochs = Convert.ToInt32(Console.ReadLine());
            Console.Write("Learn Rate:"); learnRate = Convert.ToDouble(Console.ReadLine());

            // artificial; in realistic scenarios you'd read from a text file
            int numItems = 46641*2;
            Console.WriteLine("\nProgrammatically reading " + numItems + " training data items");

            double[][] trainData = new double[numItems][];

            //讀取所有特徵資料, 並標準化
            StreamReader[] arrSR = new StreamReader[] {
                                new StreamReader(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFaceMaxPoolingPixel_64x64_FlipH_EquHist.csv"),
                                //new StreamReader(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFaceMaxPoolingPixel_64x64_EquHist.csv"),
                                //new StreamReader(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFaceMaxPoolingPixel_64x64_FlipH.csv"),
                                new StreamReader(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFaceMaxPoolingPixel_64x64.csv") };
            {
                string[][] RawTrainData = new string[trainData.Length][];
                for (int k = 0; k < arrSR.Length; k++)
                {
                    string Line;
                    for (int i = 0; (Line = arrSR[k].ReadLine()) != null; i++)
                    {
                        string[] ReadLine_Array = Line.Split(',');
                        string _PSPI = ReadLine_Array[1];
                        string[] PixelValue4096_PSPI = new string[numInput + 1];

                        for (int j = 0; j < numInput; j++)
                        {
                            PixelValue4096_PSPI[j] = ReadLine_Array[j + 2];
                        }

                        PixelValue4096_PSPI[numInput] = _PSPI;
                        RawTrainData[k * 46641 + i] = PixelValue4096_PSPI;
                    }
                }

                string[] colTypes = new string[numInput + numOutput];
                for (int i = 0; i < colTypes.Length; i++) colTypes[i] = "numeric";
                Standardizer stder = new Standardizer(RawTrainData, colTypes);
                trainData = stder.StandardizeAll(RawTrainData);
            }


            //Console.WriteLine("\nTraining data:\n");
            //Show.ShowMatrix(trainData, 3, 4, true);

            //呈現視覺化資料
            //視覺化.ShowPlot(trainData);

            Console.WriteLine("\nCreating a " + numInput + "-" + numHidden + "-" + numOutput + " regression neural network");
            Console.WriteLine("Using tanh hidden layer activation");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput, rndSeed);

            //Per-training
            //Console.WriteLine("\nPer-Training...\n");
            //double[] perTrainData = new double[1024 + 1]; for (int i = 0; i < perTrainData.Length; i++) perTrainData[i] = 0.01;
            //nn.Train(new double[1][]{perTrainData}, 1000, learnRate, momentum);

            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F4"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F4"));

            Console.WriteLine("\nStarting training (using stochastic back-propagation)");
            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum);
            Console.WriteLine("Finished training");
            //Console.WriteLine("\nFinal neural network model weights:\n");
            //ShowVector(weights, 4, 8, true);

            //讀取所有驗證資料, 並標準化
            double[][] inputVector = new double[numItems][];
            using (StreamReader SR = new StreamReader(@"C:\Users\deo\Google 雲端硬碟\碩士論文\實驗用資料庫\PoolingTestSet\UNBC_PainFaceMaxPoolingPixel_64x64_FlipH.csv"))
            {
                string Line;
                string[][] RawVerificationData = new string[trainData.Length][];

                for (int i = 0; (Line = SR.ReadLine()) != null; i++)
                {
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
            for(int target=15,i=0;i<inputVector.Length && -1!=target; i++)
            {
                if (target == inputVector[i][inputVector[i].Length-1]) //inputVector[i][last] = PSPI
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
            double COVxy = 0, Sx = 0, Sy = 0;

            for (int i = 0; i < inputVector.Length; i++)
            {//算出實際和預測的平均值
                PredictedPSPIAvg += nn.ComputeOutputs(inputVector[i])[0];
                ActualPSPIAvg += inputVector[i][inputVector[i].Length - 1];
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
            CORR = COVxy / Math.Pow(Sx * Sy, 0.5);
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
