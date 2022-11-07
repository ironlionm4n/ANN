using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArtificialNeuralNetwork
{
    // number of inputs coming in from the start
    public int NumberInputs;

    // number of output nodes
    public int NumberOutputs;

    // number of layers between input and output layers
    public readonly int NumberHiddenLayers;

    // number of neurons in a hidden layer
    public int NumberNeuronsPerHiddenLayer;

    // learning rate
    public double alpha = 0.1f;

    // collection of all the layers and the neurons they contain
    private List<Layer> layers = new List<Layer>();

    public ArtificialNeuralNetwork(int numberOfInputs, int numberOfOutputs, int numberOfHiddenLayers, int numberNodesPerHiddenLayer)
    {
        NumberInputs = numberOfInputs;
        NumberOutputs = numberOfOutputs;
        NumberHiddenLayers = numberOfHiddenLayers;
        NumberNeuronsPerHiddenLayer = numberNodesPerHiddenLayer;

        // if true there are hidden layers
        if (NumberHiddenLayers > 0)
        {
            // add in the input layer, numberInputs is how many inputs coming into each neuron, then the total number of neurons
            layers.Add(new Layer(NumberNeuronsPerHiddenLayer, NumberInputs));
            // create all of the hidden layers with equal number of neurons per hidden layer and number of neurons in each hidden layer
            for (int i = 0; i < NumberHiddenLayers - 1; i++)
            {
                layers.Add(new Layer(NumberNeuronsPerHiddenLayer, NumberNeuronsPerHiddenLayer));
            }

            // output layer
            layers.Add(new Layer(NumberOutputs, NumberInputs));
        }
        // no hidden layers, just need an output layer
        else
        {
            layers.Add(new Layer(NumberOutputs, NumberInputs));
        }
    }

    // returns the results of the feed forward
    public List<double> FeedForward(List<double> inputValues, List<double> desiredOutputs)
    {
        // list of inputs and outputs to keep track of each neuron
        List<double> inputs = new List<double>(inputValues);
        List<double> outputs = new List<double>();
        //Debug.Log($"inputs.Count: {inputs.Count}, NumberInputs: {NumberInputs}, outputs.Count: {outputs.Count}, inputValues.Count: {inputValues.Count}");
        if (inputs.Count != NumberInputs)
        {
            Debug.Log("Error: Number of inputs must be " + NumberInputs);
            return outputs;
        }

        // loop through each hidden layer and then the output layer
        for (int i = 0; i < NumberHiddenLayers + 1; i++)
        {
            // not dealing with the input layer
            if (i > 0)
            {
                // set the inputs of the following layer to the outputs from the previous layer
                inputs = new List<double>(outputs);
            }

            // clear the previous layers outputs so newly calculated outputs can be added for this current layer
            outputs.Clear();
            // loop through each node in each layer
            for (int j = 0; j < layers[i].numberOfNeurons; j++)
            {
                // the summation of the inputs time the weight of the neuron
                double eachInputTimesWeight = 0;
                // clear the inputs to this neuron to be replaced with outputs from previous layer
                layers[i].neurons[j].inputs.Clear();

                // loop through each weight of each node in each layer
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    // inputs was previously assigned the outputs from the previous layer
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    // multiply the weights by the inputs and add it up
                    eachInputTimesWeight += layers[i].neurons[j].weights[k] * inputs[k];
                }

                // add the bias node for each neuron in each layer
                eachInputTimesWeight += layers[i].neurons[j].bias;
                layers[i].neurons[j].output = ActivationFunction(eachInputTimesWeight);
                outputs.Add(layers[i].neurons[j].output);
            }
        }

        BackPropagation(outputs, desiredOutputs);

        return outputs;
    }

    public List<double> Predict(List<double> inputValues)
    {
        // list of inputs and outputs to keep track of each neuron
        List<double> inputs = new List<double>(inputValues);
        List<double> outputs = new List<double>();
        //Debug.Log($"inputs.Count: {inputs.Count}, NumberInputs: {NumberInputs}, outputs.Count: {outputs.Count}, inputValues.Count: {inputValues.Count}");
        if (inputs.Count != NumberInputs)
        {
            Debug.Log("Error: Number of inputs must be " + NumberInputs);
            return outputs;
        }

        // loop through each hidden layer and then the output layer
        for (int i = 0; i < NumberHiddenLayers + 1; i++)
        {
            // not dealing with the input layer
            if (i > 0)
            {
                // set the inputs of the following layer to the outputs from the previous layer
                inputs = new List<double>(outputs);
            }

            // clear the previous layers outputs so newly calculated outputs can be added for this current layer
            outputs.Clear();
            // loop through each node in each layer
            for (int j = 0; j < layers[i].numberOfNeurons; j++)
            {
                // the summation of the inputs time the weight of the neuron
                double eachInputTimesWeight = 0;
                // clear the inputs to this neuron to be replaced with outputs from previous layer
                layers[i].neurons[j].inputs.Clear();

                // loop through each weight of each node in each layer
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    // inputs was previously assigned the outputs from the previous layer
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    // multiply the weights by the inputs and add it up
                    eachInputTimesWeight += layers[i].neurons[j].weights[k] * inputs[k];
                }

                // add the bias node for each neuron in each layer
                eachInputTimesWeight += layers[i].neurons[j].bias;
                layers[i].neurons[j].output = ActivationFunction(eachInputTimesWeight);
                outputs.Add(layers[i].neurons[j].output);
            }
        }
        
        return outputs;
    }
 

    // apply the error to all of the weights
    private void BackPropagation(List<double> outputs, List<double> desiredOutputs)
    {
        double error;
        // loop through the layers starting at the output layer
        for (int i = NumberHiddenLayers; i >= 0; i--)
        {
            // loop through each neuron in each layer
            for (int j = 0; j < layers[i].numberOfNeurons; j++)
            {
                // the output layer
                if (i == NumberHiddenLayers)
                {
                    // calculate the delta
                    error = desiredOutputs[j] - outputs[j];
                    // calculated with Delta Rule: en.wikipedia.org/wiki/Delta_rule
                    // adding up all of the error gradients in each neuron will be equal to the error (delta)
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                }
                else
                {
                    // calculate error gradient with Delta Rule
                    layers[i].neurons[j].errorGradient =
                        layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    // the total error gradient summed by each node in the previous layers
                    double errorGradientSummation = 0;
                    for (int p = 0; p < layers[i + 1].numberOfNeurons; p++)
                    {
                        errorGradientSummation +=
                            layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }

                    layers[i].neurons[j].errorGradient *= errorGradientSummation;
                }

                // updating the weights of each neuron in each layer
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    // is the current layer the output layer
                    if (i == NumberHiddenLayers)
                    {
                        // error is multiplied with the output layer
                        error = desiredOutputs[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        // error gradient is used with non-output layers
                        layers[i].neurons[j].weights[j] += alpha * layers[i].neurons[j].inputs[k] *
                                                           layers[i].neurons[j].errorGradient;
                    }
                }
                // adjust bias nodes
                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }
    
    private double ActivationFunction(double eachInputTimesWeight)
    {
        return Sigmoid(eachInputTimesWeight);
    }

    private double Sigmoid(double eachInputTimesWeight)
    {
        double k = (double) Math.Exp(-eachInputTimesWeight);
        return k / (1.0f + k);
    }
}
