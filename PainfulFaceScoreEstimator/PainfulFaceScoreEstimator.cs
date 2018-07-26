using Emgu.CV;

namespace LKY
{
    public class PainfulFaceScoreEstimator
    {
        NeuralNetwork inferenceNN;

        public PainfulFaceScoreEstimator()
        {
            inferenceNN = new NeuralNetwork(1024, 32, 1, 0);
            double[] weights = WeightVector.PainFace_1024_DA;
            inferenceNN.SetWeights(weights);
        }

        public double getInferencePSPIScore(Mat faceRect)
        {
            //Max Pooling
            double[] inputVector;

            Mat PoolingImg = Pooling.Do(faceRect, true, out inputVector);
            //使用預設的平均值和標準差,做高斯正規化

            for (int i = 0; i < 1024; i++)
            //Parallel.For(0, 1024, i =>
            {
                inputVector[i] = (inputVector[i] - WeightVector.PainFace_1024_DA_Avg[i]) / WeightVector.PainFace_1024_DA_StdErr[i];
            }
            //);

            return this.inferenceNN.ComputeOutputs(inputVector)[0];
        }
    }
}
