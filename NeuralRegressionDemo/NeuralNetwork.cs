using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;

namespace LKY
{
    public class NeuralNetwork
    {
        private int numInput; // number input nodes
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[] hiddens;
        private double[] outputs;

        private double[][] ihWeights; // input-hidden
        private double[] hBiases;

        private double[][] hoWeights; // hidden-output
        private double[] oBiases;

        private Random rnd;

        public NeuralNetwork(int numInput, int numHidden, int numOutput, int seed)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];
            this.hiddens = new double[numHidden];
            this.outputs = new double[numOutput];

            this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
            this.hBiases = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            this.oBiases = new double[numOutput];

            this.rnd = new Random(seed);
            this.InitializeWeights(); // all weights and biases
        } // ctor

        private static double[][] MakeMatrix(int rows,
          int cols, double v) // helper for ctor, Train
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = v;
            return result;
        }

        private void InitializeWeights() // helper for ctor
        {
            // initialize weights and biases to random values between 0.0001 and 0.001
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            double lo = -0.001;
            double hi = +0.001;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;  // [-0.001 to +0.001]
                                                                        //initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.SetWeights(initialWeights);
        }

        public void SetWeights(double[] weights)
        {
            // copy serialized weights and biases in weights[] array
            // to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int w = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[w++];

            for (int j = 0; j < numHidden; ++j)
                hBiases[j] = weights[w++];

            for (int j = 0; j < numHidden; ++j)
                for (int k = 0; k < numOutput; ++k)
                    hoWeights[j][k] = weights[w++];

            for (int k = 0; k < numOutput; ++k)
                oBiases[k] = weights[k++];
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];

            int w = 0;
            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    result[w++] = ihWeights[i][j];

            for (int j = 0; j < numHidden; ++j)
                result[w++] = hBiases[j];

            for (int j = 0; j < numHidden; ++j)
                for (int k = 0; k < numOutput; ++k)
                    result[w++] = hoWeights[j][k];

            for (int k = 0; k < numOutput; ++k)
                result[w++] = oBiases[k];

            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < numInput; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

            for (int j = 0; j < numHidden; ++j)  // add biases to input-to-hidden sums
                hSums[j] += this.hBiases[j];

            for (int j = 0; j < numHidden; ++j)   // apply activation
                this.hiddens[j] = HyperTan(hSums[j]); // hard-coded

            for (int k = 0; k < numOutput; ++k)   // compute h-o sum of weights * hOutputs
                for (int j = 0; j < numHidden; ++j)
                    oSums[k] += hiddens[j] * hoWeights[j][k];

            for (int k = 0; k < numOutput; ++k)  // add biases to input-to-hidden sums
                oSums[k] += oBiases[k];

            Array.Copy(oSums, this.outputs, outputs.Length);  // copy without activation

            double[] retResult = new double[numOutput]; // could define a GetOutputs 
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        }

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        public double[] Train(double[][] trainData,
          int maxEpochs, double learnRate, double momentum)
        {
            // train using back-prop
            // back-prop specific arrays
            double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weights gradients
            double[] obGrads = new double[numOutput];                   // output biases gradients

            double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weights gradients
            double[] hbGrads = new double[numHidden];                   // hidden biases gradients

            double[] oSignals = new double[numOutput];                  // signals == gradients w/o associated input terms
            double[] hSignals = new double[numHidden];                  // hidden node signals

            // back-prop momentum specific arrays 
            double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
            double[] hPrevBiasesDelta = new double[numHidden];
            double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
            double[] oPrevBiasesDelta = new double[numOutput];

            // train a back-prop style NN regression using learning rate and momentum
            int epoch = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            int errInterval = maxEpochs / 100; // interval to check validation data
            while (epoch < maxEpochs)
            {
                ++epoch;  // immediately to prevent display when 0


                //呈現訓練中途
                Random rnd = new Random();
                double[][] testinput = new double[80][];

                for (int i = 0; i < 80; ++i)
                {
                    testinput[i] = new double[] { (i*6.4)/80.0, 0 };
                    //testinput[i] = new double[] { 6.4 * rnd.NextDouble(), 0 };
                    testinput[i][1] = this.ComputeOutputs(testinput[i])[0];
                    //產生一個周期內的80個sin取樣點
                }
                      

                if ((epoch % errInterval == 0 && epoch < maxEpochs ) || 1== epoch)
                {
                    double trainErr = Error(trainData);
                    Console.WriteLine("epoch = " + epoch + "  training error = " +
                      trainErr.ToString("F4"));

                    視覺化.ShowPlot(testinput);
                    CvInvoke.WaitKey();
                }

                Shuffle(sequence); // visit each training data in random order
                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];

                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);

                    ComputeOutputs(xValues); // copy xValues in, compute outputs 

                    // indices: i = inputs, j = hiddens, k = outputs

                    // 1. compute output nodes signals (assumes constant activation)
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double derivative = 1.0; // for dummy output activation f'
                        oSignals[k] = (tValues[k] - outputs[k]) * derivative;
                    }

                    // 2. compute hidden-to-output weights gradients using output signals
                    for (int j = 0; j < numHidden; ++j)
                        for (int k = 0; k < numOutput; ++k)
                            hoGrads[j][k] = oSignals[k] * hiddens[j];

                    // 2b. compute output biases gradients using output signals
                    for (int k = 0; k < numOutput; ++k)
                        obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

                    // 3. compute hidden nodes signals
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double sum = 0.0; // need sums of output signals times hidden-to-output weights
                        for (int k = 0; k < numOutput; ++k)
                        {
                            sum += oSignals[k] * hoWeights[j][k];
                        }
                        double derivative = (1 + hiddens[j]) * (1 - hiddens[j]); // for tanh
                        hSignals[j] = sum * derivative;
                    }

                    // 4. compute input-hidden weights gradients
                    for (int i = 0; i < numInput; ++i)
                        for (int j = 0; j < numHidden; ++j)
                            ihGrads[i][j] = hSignals[j] * inputs[i];

                    // 4b. compute hidden node biases gradienys
                    for (int j = 0; j < numHidden; ++j)
                        hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

                    // == update weights and biases

                    // 1. update input-to-hidden weights
                    for (int i = 0; i < numInput; ++i)
                    {
                        for (int j = 0; j < numHidden; ++j)
                        {
                            double delta = ihGrads[i][j] * learnRate;
                            ihWeights[i][j] += delta;
                            ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta; // save for next time
                        }
                    }

                    // 2. update hidden biases
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double delta = hbGrads[j] * learnRate;
                        hBiases[j] += delta;
                        hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // 3. update hidden-to-output weights
                    for (int j = 0; j < numHidden; ++j)
                    {
                        for (int k = 0; k < numOutput; ++k)
                        {
                            double delta = hoGrads[j][k] * learnRate;
                            hoWeights[j][k] += delta;
                            hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // 4. update output node biases
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double delta = obGrads[k] * learnRate;
                        oBiases[k] += delta;
                        oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }

                } // each training item

            } // while
            double[] bestWts = this.GetWeights();
            return bestWts;
        } // Train


        private void Shuffle(int[] sequence) // an instance method
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        } // Shuffle


        private double Error(double[][] data)
        {
            // MSE == average squared error per training item
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // first numInput values in trainData
            double[] tValues = new double[numOutput]; // last numOutput values

            // walk thru each training case
            for (int i = 0; i < data.Length; ++i)
            {
                Array.Copy(data[i], xValues, numInput);
                Array.Copy(data[i], numInput, tValues, 0, numOutput); // get target value(s)
                double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights

                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / data.Length;
        } // Error

    } // class NeuralNetwork
}


