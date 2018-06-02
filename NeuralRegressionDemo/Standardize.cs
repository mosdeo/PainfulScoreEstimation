using System;
using System.Collections.Generic;

namespace LKY
{
  public class Standardizer
  {
    // numeric x-data is Gaussian normalized
    // binary x-data is (-1 +1) encoded
    // categorical x-data is 1-of-(C-1) effects-coded ( ex: [0,1] or [1,0] or [-1,-1] )
    //
    // numeric y-data is left alone
    // binary y-data is 1-of-C dummy-coded
    // categorical y-data is 1-of-C dummy-coded

    public string[] colTypes; // "numeric" or "categorical"
    public string[] subTypes; // numericX, numericY, binaryX, binaryY, categoricalX, categoricalY
    public string[][] distinctValues; // [i] = col index
    public double[] means; // of numeric columns
    public double[] stdDevs;
    public int numStandardCols; // number of columns of standardized data

    public Standardizer(string[][] rawData, string[] colTypes)
    {
      this.colTypes = new string[colTypes.Length];
      Array.Copy(colTypes, this.colTypes, colTypes.Length);

      // get distinct values in each col.
      int numCols = rawData[0].Length;
      this.distinctValues = new string[numCols][];
      for (int j = 0; j < numCols; ++j)
      {
        if (colTypes[j] == "numeric")
        {
          distinctValues[j] = new string[] { "na" }; // 'not applicable' for numeric columns
        }
        else
        {
          Dictionary<string, bool> values = new Dictionary<string, bool>();
          for (int i = 0; i < rawData.Length; ++i)
          {
            string v = rawData[i][j];
            if (values.ContainsKey(v) == false)
              values.Add(v, true);
          }
          distinctValues[j] = new string[values.Count]; // allocate
          int k = 0;
          foreach (string s in values.Keys)
          {
            distinctValues[j][k] = s;
            ++k;
          }
        }
      } // get distinct values in each col

      // compute means of numeric cols
      this.means = new double[numCols];
      for (int j = 0; j < numCols; ++j)
      {
        if (colTypes[j] == "categorical")
        {
          this.means[j] = -1.0; // dummy values for categorical columns
        }
        else
        {
          double sum = 0.0;
          for (int i = 0; i < rawData.Length; ++i)
          {
            double v = double.Parse(rawData[i][j]);
            sum += v;
          }
          this.means[j] = sum / rawData.Length;
        }
      } // compute means

      // compute standard deviations of numeric cols
      this.stdDevs = new double[numCols];
      for (int j = 0; j < numCols; ++j)
      {
        if (colTypes[j] == "categorical")
        {
          this.stdDevs[j] = -1.0; // dummy values for categorical columns
        }
        else
        {
          double ssd = 0.0; // sum of squared deviations
          for (int i = 0; i < rawData.Length; ++i)
          {
            double v = double.Parse(rawData[i][j]);
            ssd += (v - this.means[j]) * (v - this.means[j]);
          }
          this.stdDevs[j] = Math.Sqrt(ssd / rawData.Length);
        }
      } // compute standard deviations

      // compute column subTypes
      this.subTypes = new string[numCols];
      for (int j = 0; j < numCols; ++j)
      {
        if (colTypes[j] == "numeric" && j != numCols - 1) // not last column
          this.subTypes[j] = "numericX";
        else if (colTypes[j] == "numeric" && j == numCols - 1) // last column
          this.subTypes[j] = "numericY";
        else if (colTypes[j] == "categorical" && j != numCols - 1 && distinctValues[j].Length == 2)
          this.subTypes[j] = "binaryX";
        else if (colTypes[j] == "categorical" && j == numCols - 1 && distinctValues[j].Length == 2)
          this.subTypes[j] = "binaryY";
        else if (colTypes[j] == "categorical" && j != numCols - 1 && distinctValues[j].Length >= 3)
          this.subTypes[j] = "categoricalX";
        else if (colTypes[j] == "categorical" && j == numCols - 1 && distinctValues[j].Length >= 3)
          this.subTypes[j] = "categoricalY";
      }

      // compute number of columns of standardized data
      int ct = 0;
      for (int j = 0; j < numCols; ++j)
      {
        if (this.subTypes[j] == "numericX") 
          ++ct;
        else if (this.subTypes[j] == "numericY") 
          ++ct;
        else if (this.subTypes[j] == "binaryX")
          ++ct;
        else if (this.subTypes[j] == "binaryY")
          ct += 2;
        else if (this.subTypes[j] == "categoricalX")
          ct += distinctValues[j].Length - 1;
        else if (this.subTypes[j] == "categoricalY")
          ct += distinctValues[j].Length;
      }
      this.numStandardCols = ct;
    } // ctor

    private int IndexOf(int col, string catValue)
    {
      // returns the index value of a categorical value
      // if (urban, suburban, rural) then IndexOf(rural) = 2
      for (int k = 0; k < this.distinctValues[col].Length; ++k)
        if (distinctValues[col][k] == catValue)
          return k;
      return -1; // fatal error
    }

    public double[] GetStandardRow(string[] tuple)
    {
      // ex: "30 male 38000.00 suburban democrat" ->
      // [ -0.25 -1.0 -0.75 (1.0  0.0) (0.0  0.0  0.0  1.0) ]
      double[] result = new double[this.numStandardCols];
      
      int p = 0; // ptr into result data
      for (int j = 0; j < tuple.Length; ++j)
      {
        if (this.subTypes[j] == "numericX")
        {
          double v = double.Parse(tuple[j]);
          result[p++] = (v - this.means[j]) / this.stdDevs[j]; // Gaussian normalize
        }
        else if (this.subTypes[j] == "numericY")
        {
          double v = double.Parse(tuple[j]);
          result[p++] = v; // leave alone (regression problem)
        }
        else if (this.subTypes[j] == "binaryX")
        {
          string v = tuple[j];
          int index = IndexOf(j, v); // 0 or 1. binary x-data -> -1 +1
          if (index == 0)
            result[p++] = -1.0;
          else
            result[p++] = 1.0;
        }
        else if (this.subTypes[j] == "binaryY") // y-data is 'male' or 'female'
        {
          string v = tuple[j];
          int index = IndexOf(j, v); // 0 or 1. binary x-data -> -1 +1
          if (index == 0) { result[p++] = 0.0; result[p++] = 1.0; }
          else { result[p++] = 1.0; result[p++] = 0.0; }
        }
        else if (this.subTypes[j] == "categoricalX") // ex: x-data is 'democrat' 'republican' 'independent' 'other'
        {
          string v = tuple[j];
          int ct = distinctValues[j].Length; // ex: 4
          double[] tmp = new double[ct-1]; // [ _ _ _ ]
          int index = IndexOf(j, v); // 0, 1, 2, 3
          if (index == ct - 1) // last item goes to -1 -1 -1 (effects coding)
          {
            for (int k = 0; k < tmp.Length; ++k)
              tmp[k] = -1.0;
          }
          else
          {
            for (int k = 0; k < tmp.Length; ++k)
              tmp[k] = 0.0; // not necessary in C# . . 

            tmp[ct - index - 2] = 1.0; // a bit tricky
          }
          // copy tmp values into result
          for (int k = 0; k < tmp.Length; ++k)
            result[p++] = tmp[k];
        }
        else if (this.subTypes[j] == "categoricalY")
        {
          string v = tuple[j];
          int ct = distinctValues[j].Length; // ex: 4
          double[] tmp = new double[ct]; // [ _ _ _ _ ]
          int index = IndexOf(j, v); // 0, 1, 2, 3
          for (int k = 0; k < tmp.Length; ++k)
            tmp[k] = 0.0; // not necessary in C# . . 
          tmp[ct - index - 1] = 1.0;
          for (int k = 0; k < tmp.Length; ++k)
            result[p++] = tmp[k];
        }
      } // each j col
      return result;
    } // GetStandardRow

    public double[][] StandardizeAll(string[][] rawData)
    {
      double[][] result = new double[rawData.Length][];
      for (int i = 0; i < rawData.Length; ++i)
      {
        double[] stdRow = this.GetStandardRow(rawData[i]);
        result[i] = stdRow;
      }
      return result;
    }
  } // class Standardizer

  // ==== Display Helpers =====

  public class Helpers
  {
    public static void ShowMatrix(string[][] matrix, bool indices, bool newLine)
    {
      for (int i = 0; i < matrix.Length; ++i)
      {
        if (indices == true) Console.Write("[" + i + "] ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          Console.Write(matrix[i][j] + "  ");
        }
        Console.WriteLine("");
      }
      if (newLine == true) Console.WriteLine("");
    }

    public static void ShowMatrix(double[][] matrix, int decimals, bool indices, bool newLine)
    {
      for (int i = 0; i < matrix.Length; ++i)
      {
        if (indices == true) Console.Write("[" + i + "] ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          Console.Write(matrix[i][j].ToString("F" + decimals).PadLeft(6) + " ");
        }
        Console.WriteLine("");
      }
      if (newLine == true) Console.WriteLine("");
    }

    public static void ShowVector(double[] vector, int decimals, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
        Console.Write(vector[i].ToString("F" + decimals).PadLeft(5) + " ");
      Console.WriteLine("");
      if (newLine == true) Console.WriteLine("");
    }
  }
} // ns
