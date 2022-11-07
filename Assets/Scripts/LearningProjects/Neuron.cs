using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    // neuron is connected to each neuron in the previous/next layer
    public int numInputs;
    
    // bias to move around results
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();

    // constructor for a neuron to assign random weight between 0 and 1
    public Neuron(int numberInputs)
    {
        bias = 1f;
        numInputs = numberInputs;
        for (int i = 0; i < numberInputs; i++)
        {
            weights.Add(Random.Range(0.0000001f, .99999999f));
        }
    }
    
}
