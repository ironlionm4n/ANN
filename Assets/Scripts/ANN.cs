using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;

public class ANN : MonoBehaviour
{
    public float learningRate; // used in back-propagation to adjust weights
    [SerializeField] private int[] nodesPerLayer;
    [SerializeField] private string trainingDataFileName;
    [SerializeField] private string testDataFileName;
    private List<string[]> _linesInTrainingData;
    private int[] _trainingDataValues;
    private string _path = "";
    private float _outputExpected;
    private float _outputActual;
    private Layer _inputLayer; // 64 nodes total
    private Layer[] _hiddenLayers; // arbitrary - tweak to perfect, set weights
    private Layer _outputLayer; // 

    // initialize inputLayer, hiddenLayers, outputLayer, 

    
    private void Start()
    {
        _inputLayer = new Layer();
        _linesInTrainingData = new List<string[]>();
        _trainingDataValues = new int[65];
        _path = Application.dataPath;
        var streamReader = new StreamReader(_path + "/" + trainingDataFileName);
        Init(nodesPerLayer);
        var counter = 0;
        while (!streamReader.EndOfStream)
        {
            var lineOfDigits = streamReader.ReadLine()?.Split(',');
            _linesInTrainingData.Add(lineOfDigits);
            for (var i = 0; i < 64; i++)
            {
                _trainingDataValues[i] = int.Parse(_linesInTrainingData[counter][i]);
            }
            
            // assign expected output
            _outputExpected = int.Parse(_linesInTrainingData[counter][64]);
            ProcessData();
            counter++;
        }
    }

    private void Init(int[] nodesPerLayer)
    {
        // reading in first 64 characters from file for input
        _inputLayer.Nodes = new Node[64];
        // nodesPerLayer length represents the number of layers in the ANN
        _hiddenLayers = new Layer[nodesPerLayer.Length];

        // assign weights for each node in the input layer
        for (int i = 0; i < _inputLayer.Nodes.Length; i++)
        {
            // add weights for each node in the next layer - the first hidden layer after the input layer
            _inputLayer.Nodes[i].CurrentWeights = new float[nodesPerLayer[0]];
            _inputLayer.Nodes[i].ErrorAdjustments = new[] {0f};
            
            for (int j = 0; j < nodesPerLayer[0]; j++)
            {
                _inputLayer.Nodes[i].CurrentWeights[j] = Random.Range(0.0f, 1.0f);
            }
        }

        // each element in nodesPerLayer represents the number of nodes in that layer
        for (int i = 0; i < nodesPerLayer.Length; i++)
        {
            // create the array of nodes for each hidden layer 
            _hiddenLayers[i].Nodes = new Node[nodesPerLayer[i]];
            for (int j = 0; j < nodesPerLayer[i]; j++)
            {
                // initialize each node created previously
                _hiddenLayers[i].Nodes[j] = new Node();
                if (i != nodesPerLayer.Length - 1)
                {
                    // create array of weights for each node in each layer
                    _hiddenLayers[i].Nodes[j].CurrentWeights = new float[nodesPerLayer[i + 1]];
                    _hiddenLayers[i].Nodes[j].ErrorAdjustments = new[] {0f};
                    for (int w = 0; w < nodesPerLayer[i + 1]; w++)
                    {
                        // assign weights for each node in each layer
                        _hiddenLayers[i].Nodes[j].CurrentWeights[w] = Random.Range(0.0f, 1.0f);
                    }
                }
                else
                {
                    // for output layer weight is not needed
                    _hiddenLayers[i].Nodes[j].CurrentWeights = new[] {Random.Range(0f,1f)};
                    _hiddenLayers[i].Nodes[j].ErrorAdjustments = new float[1];
                }
            }
        }
    }
    
    // 
    public bool ProcessData()
    {
       // read input
       ReadInput(_trainingDataValues);
       // forward feed
       Debug.Log(ForwardFeed());
       // Debug.Log(_outputExpected - _outputActual);
       // back-propagate
       BackwardsPropagate();
       // return actual - expected < .5
       return true;
    }

    private void ReadInput(int[] trainingData)
    {
        for (int i = 0; i < trainingData.Length - 1; i++)
        {
            // assigning both current sum and previous sum in single line
            _inputLayer.Nodes[i].CurrentSum = _inputLayer.Nodes[i].PreviousSum = trainingData[i];
        }
    }

    // Need to find the current sum for each node in the current layer
    private void Activation(Layer currentLayer)
    {
        for (int i = 0; i < currentLayer.Nodes.Length; i++)
        {
            var denominator = (1 + Mathf.Exp(-currentLayer.Nodes[i].PreviousSum));
            currentLayer.Nodes[i].CurrentSum = (1 / denominator);
        }
    }

    private float ForwardFeed()
    {
        // i = -1 to check the input layer to hidden layer length - 1 to not access the output layer
        for (var i = -1; i < _hiddenLayers.Length - 1; i++)
        {
            // reference the appropriate layers
            Layer currentLayer; 
            Layer nextLayer; 
            
            // handle input layer case
            if (i == -1)
            {
                currentLayer = _inputLayer;
                nextLayer = _hiddenLayers[i + 1];
            }
            else
            {
                currentLayer = _hiddenLayers[i];
                nextLayer = _hiddenLayers[i + 1];
                
                // activation function
                Activation(currentLayer);
            }
            // for all the nodes in the current layer
            for (var j = 0; j < currentLayer.Nodes.Length; j++)
            {
                Node currentLayerNode = currentLayer.Nodes[j];
                // for all the current weights in the next layers node
                for (var w = 0; w < currentLayerNode.CurrentWeights.Length; w++)
                {
                    var weightedSumOfTheInputs = currentLayerNode.CurrentSum * currentLayerNode.CurrentWeights[w];
                    nextLayer.Nodes[w].PreviousSum += weightedSumOfTheInputs;
                }
            }
            if (i == -1)
            {
                _inputLayer = currentLayer;
                _hiddenLayers[i + 1] = nextLayer;
            }
            else
            {
                _hiddenLayers[i] = currentLayer;
                _hiddenLayers[i + 1] = nextLayer;
            }
        }
        Activation(_hiddenLayers[^1]);
        // only 1 node in the last hidden layer - aka output layer
        return _outputActual = _hiddenLayers[^1].Nodes[0].CurrentSum;
    }
    
    private void BackwardsPropagate()
    {
        // output errors
        OutputErrors();
        // hidden errors
        HiddenErrors();
        // input errors
        InputErrors();
    }

    private void InputErrors()
    {
        // j is the current node, i is the next layer
        for (int j = 0; j < _inputLayer.Nodes.Length; j++)
        {
            float summationOfWeightsAndDeltas = 0f;
            for (int k = 0; k < _inputLayer.Nodes[j].CurrentWeights.Length; k++)
            {
                summationOfWeightsAndDeltas += _inputLayer.Nodes[j].CurrentWeights[k] *
                                               _hiddenLayers[0].Nodes[k].ErrorAdjustments[0];
            }

            _inputLayer.Nodes[j].ErrorAdjustments[0] =
                CalculateInverseSigmoid(_inputLayer.Nodes[j].PreviousSum) * summationOfWeightsAndDeltas;
            
            for (int k = 0; k < _inputLayer.Nodes[j].CurrentWeights.Length; k++)
            {
                _inputLayer.Nodes[j].CurrentWeights[k] += _hiddenLayers[0].Nodes[k].ErrorAdjustments[0] *
                                                                   learningRate * _inputLayer.Nodes[k].CurrentSum;

                _hiddenLayers[0].Nodes[k].PreviousSum = 0f;
            }

            _inputLayer.Nodes[j].PreviousSum = 0f;
        }
    }

    // Adjust weights for the outputs
    private void OutputErrors()
    {
        Debug.Log($"hid lay node 0 prev sum: {_hiddenLayers[^1].Nodes[0].PreviousSum}");
        // Debug.Log($"Expected {_outputExpected} Actual {_outputActual}");
        var delta = CalculateInverseSigmoid(_hiddenLayers[^1].Nodes[0].PreviousSum) * (_outputExpected - _outputActual);
        Debug.Log("delta: "+delta);
        _hiddenLayers[^1].Nodes[0].ErrorAdjustments[0] = delta;
        _hiddenLayers[^1].Nodes[0].CurrentWeights[0] += learningRate * _hiddenLayers[^1].Nodes[0].CurrentSum * delta;
    }

    // Used in backwards propagation
    private float CalculateInverseSigmoid(float y)
    {
        return Mathf.Log(y / (1 - y));
    }

    private void HiddenErrors()
    {
        for (int i = _hiddenLayers.Length - 1; i > 1; i--)
        {
            // j is the current node, i is the next layer
            for (int j = 0; j < _hiddenLayers[i - 1].Nodes.Length; j++)
            {
                float summationOfWeightsAndDeltas = 0f;
                for (int k = 0; k < _hiddenLayers[i - 1].Nodes[j].CurrentWeights.Length; k++)
                {
                    summationOfWeightsAndDeltas += _hiddenLayers[i - 1].Nodes[j].CurrentWeights[k] *
                                                   _hiddenLayers[i].Nodes[k].ErrorAdjustments[0];
                } 
                
                _hiddenLayers[i - 1].Nodes[j].ErrorAdjustments[0] =
                    CalculateInverseSigmoid(_hiddenLayers[i - 1].Nodes[j].PreviousSum) * summationOfWeightsAndDeltas;
                for (int k = 0; k < _hiddenLayers[i - 1].Nodes[j].CurrentWeights.Length; k++)
                {
                    _hiddenLayers[i - 1].Nodes[j].CurrentWeights[k] += _hiddenLayers[i].Nodes[k].ErrorAdjustments[0] *
                                                                       learningRate * _hiddenLayers[i - 1].Nodes[k].CurrentSum;
                    /*Debug.Log($"hiddenlayers.nodes.currentweights: {_hiddenLayers[i-1].Nodes[j].CurrentWeights[k]} i: {i} j: {j} k: {k}");*/
                }

                _hiddenLayers[i - 1].Nodes[j].PreviousSum = 0;
            }
        }    
    }
}

public struct Node
{
    // each node will need to track the current and previous sums
    public float CurrentSum;
    public float PreviousSum; // before activation function, used for actual - expected multiplication

    // the collection of weights and adjustment values for weights
    public float[] CurrentWeights;
    public float[] ErrorAdjustments;
}

// Each layer has a collection of nodes 
public struct Layer
{
    public Node[] Nodes;
}