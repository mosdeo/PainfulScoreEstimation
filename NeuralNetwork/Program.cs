using System;
using System.Threading.Tasks;
using Emgu.CV;
using LKY;

namespace NeuralRegression
{
    class NeuralRegressionProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network regression demo\n");
            Console.WriteLine("Goal is to predict the sin(x)");

            // artificial; in realistic scenarios you'd read from a text file
            int numItems = 80;
            Console.WriteLine("\nProgrammatically generating " +
              numItems + " training data items");

            double[][] trainData = new double[numItems][];
            Random rnd = new Random();

            for (int i = 0; i < numItems; ++i)
            {
                double x = 6.4 * rnd.NextDouble(); // [0 to 2PI]
                double sx = Math.Sin(x);
                trainData[i] = new double[] { x, sx };
                //產生一個周期內的80個sin取樣點
            }

            Console.WriteLine("\nTraining data:\n");
            Show.ShowMatrix(trainData, 3, 4, true);

            //呈現視覺化資料
            視覺化.ShowPlot(trainData);
            //CvInvoke.WaitKey(1000);
            CvInvoke.WaitKey();

            int numInput = 1; // usually more
            int numHidden = 4;
            int numOutput = 1; // usual for regression
            int rndSeed = 0;

            Console.WriteLine("\nCreating a " + numInput + "-" +
              numHidden + "-" + numOutput + " regression neural network");
            Console.WriteLine("Using tanh hidden layer activation");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput, rndSeed);

            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.001;
            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F4"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F4"));

            Console.WriteLine("\nStarting training (using stochastic back-propagation)");
            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum);
            Console.WriteLine("Finished training");
            Console.WriteLine("\nFinal neural network model weights:\n");
            Show.ShowVector(weights, 4, 8, true);

            double[] y = nn.ComputeOutputs(new double[] { Math.PI });
            Console.WriteLine("\nActual sin(PI)       =  0.0   Predicted =  " + y[0].ToString("F6"));

            y = nn.ComputeOutputs(new double[] { Math.PI / 2 });
            Console.WriteLine("\nActual sin(PI / 2)   =  1.0   Predicted =  " + y[0].ToString("F6"));

            y = nn.ComputeOutputs(new double[] { 3 * Math.PI / 2.0 });
            Console.WriteLine("\nActual sin(3*PI / 2) = -1.0   Predicted = " + y[0].ToString("F6"));

            y = nn.ComputeOutputs(new double[] { 6 * Math.PI });
            Console.WriteLine("\nActual sin(6*PI)     =  0.0   Predicted =  " + y[0].ToString("F6"));

            Console.WriteLine("\nEnd demo\n");
            Console.ReadLine();
        } // Main
    }  // Program
}
