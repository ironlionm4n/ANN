using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    // number of neurons in a layer instance
    public int numberOfNeurons;
    // collection of neurons in a layer instance
    public List<Neuron> neurons = new List<Neuron>();
    
    // layer constructor that creates all of the neurons for this layer instance
    // number of neuron inputs is the number of neurons in the previous layer
    public Layer(int numNeurons, int numNeuronInputs)
    {
        numberOfNeurons = numNeurons;
        for (int i = 0; i < numNeurons; i++)
        {
            neurons.Add(new Neuron(numNeuronInputs));
        }
    }
}
