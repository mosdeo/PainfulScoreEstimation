using System;
namespace LKY
{
  public class NeuralNetwork2
  {
    private int numInput; // number input nodes
    private int numHidden;
    private int numOutput;

    private double[] inputs;
    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hiddens;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;
    private double[] outputs;

    private Random rnd;

    public NeuralNetwork2(int numInput, int numHidden, int numOutput, int seed)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden);
      this.hBiases = new double[numHidden];
      this.hiddens = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput);
      this.oBiases = new double[numOutput];
      this.outputs = new double[numOutput];

      this.rnd = new Random(seed);
      this.InitializeWeights(); // all weights and biases
    } // ctor

    private static double[][] MakeMatrix(int rows,
      int cols) // helper for ctor, Train
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      return result;
    }

    private void InitializeWeights() // helper for ctor
    {
      // initialize weights and biases to random values between -0.001 and +0.001
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      double lo = -0.001;
      double hi = 0.001;
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
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

      int p = 0; // points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[p++];
      for (int j = 0; j < numHidden; ++j)
        hBiases[j] = weights[p++];
      for (int j = 0; j < numHidden; ++j)
        for (int k = 0; k < numOutput; ++k)
          hoWeights[j][k] = weights[p++];
      for (int k = 0; k < numOutput; ++k)
        oBiases[k] = weights[p++];
    }

    public double[] GetWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] result = new double[numWeights];
      int p = 0;
      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          result[p++] = ihWeights[i][j];
      for (int j = 0; j < numHidden; ++j)
        result[p++] = hBiases[j];
      for (int j = 0; j < numHidden; ++j)
        for (int k = 0; k < numOutput; ++k)
          result[p++] = hoWeights[j][k];
      for (int k = 0; k < numOutput; ++k)
        result[p++] = oBiases[k];
      return result;
    }

    public double[] ComputeOutputs(double[] xValues)
    {
      double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
      double[] oSums = new double[numOutput]; // output nodes sums

      for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
        this.inputs[i] = xValues[i];
      // note: no need to copy x-values unless you implement a ToString.
      // more efficient is to simply use the xValues[] directly.


      for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

      for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
        hSums[i] += this.hBiases[i];

      for (int i = 0; i < numHidden; ++i)   // apply activation
        this.hiddens[i] = HyperTan(hSums[i]); // hard-coded

      for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hiddens[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
        oSums[i] += oBiases[i];

      double[] softOut = Softmax(oSums); // all outputs at once for efficiency
      Array.Copy(softOut, outputs, softOut.Length);

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

    private static double[] Softmax(double[] oSums)
    {
      // does all output nodes at once so scale
      // doesn't have to be re-computed each time

      // if (oSums.Length < 2) throw . . . 

      double[] result = new double[oSums.Length];

      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i]);

      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i]) / sum;

      return result; // now scaled so that xi sum to 1.0
    }

    public double[] Train(double[][] trainData,
      int maxEpochs, double learnRate,
      double momentum, double decay, bool progress)
    {
      // train using back-prop
      // back-prop specific arrays
      double[][] hoGrads = MakeMatrix(numHidden, numOutput); // hidden-to-output weights gradients
      double[] obGrads = new double[numOutput];                   // output biases gradients

      double[][] ihGrads = MakeMatrix(numInput, numHidden);  // input-to-hidden weights gradients
      double[] hbGrads = new double[numHidden];                   // hidden biases gradients

      double[] oSignals = new double[numOutput];                  // output signals - gradients w/o associated input terms
      double[] hSignals = new double[numHidden];                  // hidden node signals

      // back-prop momentum specific arrays 
      double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
      double[] hPrevBiasesDelta = new double[numHidden];
      double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
      double[] oPrevBiasesDelta = new double[numOutput];

      // train a back-prop style NN classifier using learning rate and momentum
      int epoch = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // target values

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int errInterval = maxEpochs / 10; // interval to check validation data
      while (epoch < maxEpochs)
      {
        ++epoch;

        //if (progress == true && epoch % errInterval == 0 && epoch < maxEpochs)
        {
          double trainErr = Error(trainData);
          Console.WriteLine("epoch = " + epoch + "  training error = " +
            trainErr.ToString("F4"));
          //Console.ReadLine();
        }

        Shuffle(sequence); // visit each training data in random order
        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];
          Array.Copy(trainData[idx], xValues, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues); // copy xValues in, compute outputs 

          // indices: i = inputs, j = hiddens, k = outputs

          // 1. compute output nodes signals (assumes softmax)
          for (int k = 0; k < numOutput; ++k)
            oSignals[k] = (tValues[k] - outputs[k]) * (1 - outputs[k]) * outputs[k];

          // 2. compute hidden-to-output weights gradients using output signals
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              hoGrads[j][k] = oSignals[k] * hiddens[j];
            }
          }

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
            hSignals[j] = (1 + hiddens[j]) * (1 - hiddens[j]) * sum;  // assumes tanh
          }

          // 4. compute input-hidden weights gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * inputs[i];

          // 4b. compute hidden node biases gradienys
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

          // == update weights and biases

          // update input-to-hidden weights
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta;
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
              ihWeights[i][j] -= (decay * ihWeights[i][j]); // weight decay
              ihPrevWeightsDelta[i][j] = delta; // save for next time
            }
          }

          // update hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
            hBiases[j] += hPrevBiasesDelta[j] * momentum;
            // weight decay not normally used on biases
            hPrevBiasesDelta[j] = delta;
          }

          // update hidden-to-output weights
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
              hoWeights[j][k] -= (decay * hoWeights[j][k]); // weight decay
              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          // update output node biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            // weight decay not normally used on biases
            oPrevBiasesDelta[k] = delta;
          }

        } // each training item

      } // while
      double[] bestWts = GetWeights();
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
      // mean (average) squared error per training item
      double sumSquaredError = 0.0;
      double[] xValues = new double[numInput]; // first numInput values in trainData
      double[] tValues = new double[numOutput]; // last numOutput values

      // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, numInput);
        Array.Copy(data[i], numInput, tValues, 0, numOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
        for (int j = 0; j < numOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / data.Length;
    } // Error

    public double Accuracy(double[][] data)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, numInput); // get x-values
        Array.Copy(data[i], numInput, tValues, 0, numOutput); // get t-values
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
        int tMaxIndex = MaxIndex(tValues);

        if (maxIndex == tMaxIndex)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    private static int MaxIndex(double[] vector) // helper for Accuracy()
    {
      // index of largest value
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }

  } // NeuralNetwork


} // ns
