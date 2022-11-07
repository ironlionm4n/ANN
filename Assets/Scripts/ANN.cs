using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;

public class ANN : MonoBehaviour
{
    [SerializeField] private double learningRate = 0.2;
    [SerializeField] private string trainingDataFileName;
    [SerializeField] private string testingDataFileName;
    private string trainingDataFilePath;
    private string testingDataFilePath;
    double numberCorrect = 0;
    double numberTested = 0;
    int numberOfLayers = 5;
    List<double>[] layers;
    List<List<double>>[] weights;
    double[] changeInAnswer;
    List<double>[] changeInWeight;
    StreamReader trainingDataReader;
    StreamReader testingDataReader;
    string trainingDataText;
    string[] currentInputString;
    double[] currentInput;
    private double sumWeightsValues = 0;
    private double highestOutputNode = 0;
    private double highestOutputFound = 0;

    // Start is called before the first frame update
    void Start()
    {
        trainingDataFilePath = $"{Application.dataPath}/"+"optdigits_train.txt";
        testingDataFilePath = $"{Application.dataPath}/"+"optdigits_test.txt";
        trainingDataReader = new StreamReader(trainingDataFilePath);
        testingDataReader = new StreamReader(testingDataFilePath);
        string path = $"{Application.dataPath}/testAccuracy.txt";
        var sr = File.CreateText(path);
        Init();
        for (var i = 0; i < 5; i++)
        {
            TrainNeuralNet();
            TestNeuralNet();
            //Debug.Log($"NN was right " + ((numberCorrect) / numberTested) * 100 + $"%, numberCorrect: {numberCorrect}, numberTested: {numberTested}");
            sr.WriteLine($"{numberCorrect},{numberTested},{numberCorrect/numberTested}");
        }
        sr.Close();
    }

    // g
    public double Sigmoid(double val)
    {
        return 1 / (1 + (Math.Exp(-val)));
    }

    // g'
    public double SigmoidDerivative(double val)
    {
        // return Sigmoid(val) * (1 - Sigmoid(val));
        return Math.Log(Sigmoid(val) / (1 - Sigmoid(val)));
    }

    public double InverseSigmoid(double val)
    {
        return Math.Log(Sigmoid(val) / (1 - Sigmoid(val)));
    }

    private void Init()
    {
        // read in the first line of the training data
        trainingDataText = trainingDataReader.ReadLine();
        // split the texts via a comma delimiter 
        currentInputString = trainingDataText.Split(",");
        // initialize the size of the input array to be the length of the current input string
        currentInput = new double[currentInputString.Length];
        // initialize layers, weights, change in weights
        layers = new List<double>[numberOfLayers];
        weights = new List<List<double>>[numberOfLayers];
        changeInWeight = new List<double>[numberOfLayers];
        // delta between expected and actual
        changeInAnswer = new double[10];

        // initializes all of the layers
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new List<double>();
        }


        for (int i = 0; i < currentInputString.Length; i++)
        {
            currentInput[i] = double.Parse(currentInputString[i]);
            if (i == currentInputString.Length - 1)
            {
                break;
            }
            // add the input values to each node in the input layer
            layers[0].Add(currentInput[i]);
        }

        // add all of the hidden nodes
        int numberOfNodes = layers[0].Count / 2;
        for (int i = 1; i < layers.Length-1; i++)
        {
            for (int j = 0; j < numberOfNodes - 1; j++)
            {
                layers[i].Add(0);
            }

            //Add the bias node for each layer
            layers[i].Add(1);
        }

        // 10 possible output nodes
        for (int i = 0; i < 10; i++)
        {
            // index from end syntax
            layers[^1].Add(0);
        }

        // initializes weights and change in weights for each layer
        for (int i = 1; i < layers.Length; i++)
        {
            weights[i] = new List<List<double>>();
            changeInWeight[i] = new List<double>();

            // for each node in the neural net layers
            for (int j = 0; j < layers[i].Count; j++)
            {
                weights[i].Add(new List<double>());
                changeInWeight[i].Add(0);
                
                // for each edge to this node assign a random weight between 0 and 1
                for (int k = 0; k < layers[i - 1].Count; k++)
                {
                    weights[i][j].Add(Random.Range(0.0001f, .99999f));
                }
            }
        }
    }

    private void TrainNeuralNet()
    {
        trainingDataReader = new StreamReader(trainingDataFilePath);
        // trainingDataText = trainingDataLine.ReadLine();
        
        // continue forward feeding until the training data has been selected
        while (trainingDataText != null )
        {
           // Debug.Log("New Input\n");
            /*currentInputString = trainingDataText.Split(",");
            currentInput = new double[currentInputString.Length];*/

            ForwardFeed();

            // find the highest output node
            highestOutputFound = layers[^1][0];
            for (int i = 0; i < layers[^1].Count; i++)
            {
                // switch the highest output found with the new found highest output
                if (highestOutputFound < layers[^1][i])
                {
                    highestOutputFound = layers[^1][i];
                    highestOutputNode = i;
                }
            }
            

            // if the output is the correct then dont backwards propagate 
            /*if (highestOutputNode == currentInput[^1])
            {
                //Debug.Log($"Answer was {highestOutputNode} highestOutputFound {highestOutputFound}");
            }*/
            
            
            // if the neural net did not find the correct answer back-prop
            if(highestOutputNode != currentInput[^1])
            {
                Debug.Log($"Answer was not {highestOutputNode} answer was {currentInput[^1]}");
                BackwardsPropagation();
            }

            trainingDataText = trainingDataReader.ReadLine();
            currentInputString = trainingDataText?.Split(",");
            if(currentInputString != null)
                currentInput = new double[currentInputString.Length];
        }
    }
    
    private void ForwardFeed()
    {
        for (int i = 0; i < currentInputString.Length - 1; i++)
        {
            currentInput[i] = double.Parse(currentInputString[i]);
            // Input layer
            layers[0][i] = currentInput[i];
        }

        // for each layer after input
        for (int i = 1; i < layers.Length; i++)
        {
            // for each node in each layer
            for (int j = 0; j < layers[i].Count; j++)
            {
               sumWeightsValues = 0;

                // for each edge to the current node from the previous layer
                for (int k = 0; k < weights[i].Count; k++)
                {
                    sumWeightsValues += (weights[i][j][k] * layers[i - 1][k]);
                }

                sumWeightsValues += layers[i][^1];

                // prevent bias node value from being overwritten
                if (j == layers[i].Count - 1 && i != layers.Length - 1)
                {
                    break;
                }
                layers[i][j] = Sigmoid(sumWeightsValues);
            }
        }
    }

    private void BackwardsPropagation()
    {
        
        /*// set the output nodes that are not the correct output to 0
        for (int i = 0; i < layers[^1].Count; i++)
        {
            if (i != highestOutputNode)
            {
                layers[^1][i] = 0;
            }
        }*/
        
        //Calculate change in answer for each node
        for (int i = 0; i < layers[^1].Count; i++)
        {
            //If i is equal to the answer, use 1 as the expected result
            //Else use 0 as the expected result
            //Debug.Log($"i: {i}, currentInput[^1]: {currentInput[^1]}");
            if (i == currentInput[^1])
            {
                // the change in answer will be positive if we like what we hear from the neural nets pathway
                changeInAnswer[i] = InverseSigmoid(layers[^1][i]) * (1 - layers[^1][i]);
                changeInWeight[layers.Length - 1][i] = changeInAnswer[i];
            }
            else
            {
                changeInAnswer[i] = InverseSigmoid(layers[^1][i]) * (0 - layers[^1][i]);
                changeInWeight[layers.Length - 1][i] = changeInAnswer[i];
            }
        }


        // from last hidden layer to input layer update the weights
        for (int i = layers.Length - 2; i > 0; i--)
        {
            // go through each node in the layer
            for (int j = 0; j < layers[i].Count; j++)
            {
                // for each node in the previous layer
                for (int k = 0; k < layers[i + 1].Count; k++)
                {
                    double value = 0;
                    //For each edge to the current node
                    for (int l = 0; l < layers[^1].Count; l++)
                    {
                        value += changeInWeight[i + 1][k] * weights[i][j][l];
                    }

                    changeInWeight[i][j] *= (SigmoidDerivative(layers[i][j]) * value);
                }
            }
        }


        // changes each weight of each node in each layer
        for (int i = 1; i < layers.Length; i++)
        {
            for (int j = 0; j < weights[i].Count; j++)
            {
                for (int k = 0; k < weights[i][j].Count; k++)
                {
                    weights[i][j][k] = weights[i][j][k] + (learningRate * layers[i][j] * changeInWeight[i][j]);
                    // Debug.Log($"295 weights[i]j[j][k]: {weights[i][j][k]}");
                    /*
                    // keep the weight from going below 0
                    if (weights[i][j][k] < 0)
                    {
                        weights[i][j][k] = 0;
                    }*/
                }
            }
        }
    }



    private void TestNeuralNet()
    {
        testingDataReader = new StreamReader(testingDataFilePath);
        trainingDataText = testingDataReader.ReadLine();

        // testing the ANN
        while(trainingDataText != null)
        {
            numberTested++;
            currentInputString = trainingDataText.Split(",");
            currentInput = new double[currentInputString.Length];

            for (int i = 0; i < currentInput.Length; i++)
            {
                currentInput[i] = double.Parse(currentInputString[i]);

                // the last character in the input string is the correct answer
                if (i == currentInputString.Length - 1)
                {
                    break;
                }
                layers[0][i] = currentInput[i];
            }

            // for each layer starting with the first hidden layer
            for (int i = 1; i < layers.Length; i++)
            {
                // for each node in each layer layer
                for (int j = 0; j < layers[i].Count; j++)
                {
                    double value = 0;

                    //For each edge to the current node
                    for (int k = 0; k < weights[i].Count; k++)
                    {
                        value += (weights[i][j][k] * layers[i - 1][k]);
                    }

                    // prevent the bias node's value from being overwritten
                    if (j == layers[i].Count - 1 && i != layers.Length - 1)
                    {
                        break;
                    }
                    //Debug.Log(value);
                    layers[i][j] = Sigmoid(value);
                    // Debug.Log(layers[l][node]);
                }
            }


            double highestNode = 0;
            // highest value is first set to the first node of the output layer
            double highestValue = layers[^1][0];
            
            // check and update any new higher values found
            for (int i = 0; i < layers[^1].Count; i++)
            {
                if (highestValue < layers[^1][i])
                {
                    highestValue = layers[^1][i];
                    highestNode = i;
                }
            }

            // if the highest node found matches the expected answer then increment number correct
            if (highestNode == currentInput[currentInput.Length - 1])
            {
                numberCorrect++;
            }

            //Debug.Log($"NN was right " + ((numberCorrect) / numberTested) * 100 + $"%, numberCorrect: {numberCorrect}");
            trainingDataText = testingDataReader.ReadLine();
        }
    }
}