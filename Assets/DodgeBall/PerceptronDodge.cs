using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Random = UnityEngine.Random;

public class PerceptronDodge : MonoBehaviour
{
	[SerializeField] private GameObject aiAgent;
	private List<TrainingSet> ts = new List<TrainingSet>();
	double[] weights = {0,0};
	double bias = 0;
	double totalError = 0;


	private void Start()
	{
		InitializeWeights();
	}

	private void Update()
	{
		if (Input.GetKeyDown(KeyCode.Space))
		{
			InitializeWeights();
			ts.Clear();
		}

		if (Input.GetKeyDown("s"))
		{
			SaveWeights();
		}

		if (Input.GetKeyDown("l"))
		{
			LoadWeights();
		}
		
	}

	public void FeedInput(double input1, double input2, double output)
	{
		double calculatedOutput = CalcOutput(input1, input2);
		Debug.Log($"calculated Output: {calculatedOutput}");
		if (calculatedOutput == 0)
		{
			aiAgent.GetComponent<Animator>().SetTrigger("Crouch");
			aiAgent.GetComponent<Rigidbody>().isKinematic = false;
		}
		else
		{
			aiAgent.GetComponent<Rigidbody>().isKinematic = true;
		}
		
		// learn from it for next time
		TrainingSet set = new TrainingSet();
		set.input = new double[2] {input1, input2};
		set.output = output;
		ts.Add(set);
		Train();
	}
	
	double DotProductBias(double[] v1, double[] v2) 
	{
		if (v1 == null || v2 == null)
			return -1;
	 
		if (v1.Length != v2.Length)
			return -1;
	 
		double d = 0;
		for (int x = 0; x < v1.Length; x++)
		{
			d += v1[x] * v2[x];
		}

		d += bias;
	 
		return d;
	}

	double CalcOutput(int i)
	{
		return(ActivationFunction(DotProductBias(weights,ts[i].input)));
	}

	double CalcOutput(double i1, double i2)
	{
		double[] inp = new double[] {i1, i2};
		return(ActivationFunction(DotProductBias(weights,inp)));
	}

	double ActivationFunction(double dp)
	{
		if(dp > 0) return (1);
		return(0);
	}

	void InitializeWeights()
	{
		for(int i = 0; i < weights.Length; i++)
		{
			weights[i] = Random.Range(-1.0f,1.0f);
		}
		bias = Random.Range(-1.0f,1.0f);
	}

	void UpdateWeights(int j)
	{
		double error = ts[j].output - CalcOutput(j);
		totalError += Mathf.Abs((float)error);
		for(int i = 0; i < weights.Length; i++)
		{			
			weights[i] = weights[i] + error*ts[j].input[i]; 
		}
		bias += error;
	}

	void Train()
	{
		for(int t = 0; t < ts.Count; t++)
		{
			UpdateWeights(t);
		}
	}
	
	void LoadWeights()
	{
		string path = Application.dataPath + "/weights.txt";
		if(File.Exists(path))
		{
			var sr = File.OpenText(path);
			string line = sr.ReadLine();
			string[] w = line.Split(',');
			weights[0] = System.Convert.ToDouble(w[0]);
			weights[1] = System.Convert.ToDouble(w[1]);
			bias = System.Convert.ToDouble(w[2]);
			Debug.Log("loading");
			sr.Close();
		}
	}

	void SaveWeights()
	{
		string path = Application.dataPath + "/weights.txt";
		var sr = File.CreateText(path);
		sr.WriteLine (weights[0] + "," + weights[1] + "," + bias);
		sr.Close();
	}
}