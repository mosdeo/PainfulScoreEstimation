using System;
using System.Threading.Tasks;
using Emgu.CV;
using LKY;

namespace NeuralRegression
{
    class NeuralRegressionProgram
    {
        //  1 + 4 = 5
        //  2 + 5 = 12
        //  3 + 6 = 21
        //  8 + 11 = ?

        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network regression demo\n");
            Console.WriteLine("Goal is to predict the sin(x)");

            //訓練資料
            string[][] strTrainData = new string[][] { new string[] { "1", "4", "5" },
                                                    new string[] { "2", "5", "12" },
                                                    new string[] { "3", "6", "21" } };
            double[][] trainData = new double[][] { new double[] { 1, 4, 5 },
                                                    new double[] { 2, 5, 12 },
                                                    new double[] { 3, 6, 21 } };
            Standardizer s = new Standardizer(strTrainData, new string[] { "numeric", "numeric", "numeric" });
            trainData = s.StandardizeAll(strTrainData);


            //測試資料
            string[][] strTestData = new string[][] {   new string[] { "0", "3" ,"0"},
                                                        new string[] { "2.5", "5.5" ,"0"},
                                                        new string[] { "8", "11" ,"0"} };
            double[][] testData = new double[][] {  new double[] { 0, 3 ,0},
                                                    new double[] { 2.5, 5.5 ,0},
                                                    new double[] { 8, 11 ,0} };
            testData = s.StandardizeAll(strTestData);


            //類神經網路規格參數
            int numInput = 2; // usually more
            int numHidden = 100;
            int numOutput = 1; // usual for regression
            int rndSeed = 0;

            Random rnd = new Random(1);



            Console.WriteLine("\nTraining data:\n");
            Show.ShowMatrix(trainData, 3, 4, true);

            //呈現視覺化資料
            //視覺化.ShowPlot(trainData);
            CvInvoke.WaitKey(1000);

            Console.WriteLine("\nCreating a " + numInput + "-" +
              numHidden + "-" + numOutput + " regression neural network");
            Console.WriteLine("Using tanh hidden layer activation");

            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput, rndSeed);

            int maxEpochs = 3000;
            double learnRate = 0.008;
            double momentum = 0.001;
            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F4"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F4"));

            Console.WriteLine("\nStarting training (using stochastic back-propagation)");
            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum);
            Console.WriteLine("Finished training");
            Console.WriteLine("\nFinal neural network model weights:\n");
            Show.ShowVector(weights, 4, 8, true);

            double[] y = nn.ComputeOutputs(testData[0]);

            foreach (double[] input in testData)
            {
                Console.WriteLine("\n {0} + {1} = {2} ",input[0], input[1], nn.ComputeOutputs(input)[0].ToString("F6"));
            }

            Console.WriteLine("\nEnd demo\n");
            Console.ReadLine();
        } // Main
    }  // Program
}
