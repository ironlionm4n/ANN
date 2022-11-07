using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class Brain : MonoBehaviour
{
    private ArtificialNeuralNetwork ann;
    private double sumSquareError = 0;
    private string trainingDataFilePath;
    private string testingDataFilePath;
    private List<List<double>> setOfTrainingValues = new List<List<double>>();
    private List<List<double>> setOfTrainingAnswers = new List<List<double>>();
    private List<List<double>> setOfTestingAnswers = new List<List<double>>();
    private List<List<double>> setOfTestingValues = new List<List<double>>();
    private List<List<double>> trainingResults = new List<List<double>>();

    private void Start()
    {
        var numberGuessedCorrectly = 0;
        var numberOfTotalGuesses = 0;
        // create the Ann
        ann = new ArtificialNeuralNetwork(64, 10, 3, 64);
        // store the paths to the training and test data
        trainingDataFilePath = $"{Application.dataPath}/" + "optdigits_train.txt";
        testingDataFilePath = $"{Application.dataPath}/" + "optdigits_test.txt";
        // get the training data
        ParseDataFiles(trainingDataFilePath, true);
        // get the training data
        ParseDataFiles(testingDataFilePath, false);

        for (int e = 0; e < 100; e++)
        {


            // train the ann
            for (int i = 0; i < setOfTrainingValues.Count; i++)
            {
                trainingResults.Add(ann.FeedForward(setOfTrainingValues[i], setOfTrainingAnswers[i]));
            }

            //var sw = new StreamWriter($"{Application.dataPath}/trainingResults.txt");
            /*foreach (var result in trainingResults)
            {
                var resultString = "";
                foreach (var val in result)
                {
                    resultString += val.ToString() + ",";
                }
                resultString += "\n";
                sw.WriteLine(resultString);
            }*/
            for (int i = 0; i < setOfTestingValues.Count; i++)
            {
                numberOfTotalGuesses++;
                var predictedOutputs = ann.Predict(setOfTestingValues[i]);
                Debug.Log($"predictedOutputs.Count " + predictedOutputs.Count);
                var numberPredicted = FindHighestElement(predictedOutputs);
                var actualAnswer = FindHighestElement(setOfTestingAnswers[i]);
                if (numberPredicted == actualAnswer)
                {
                    numberGuessedCorrectly++;
                }
            }

            Debug.Log(
                $"NumberGuessedCorrectly: {numberGuessedCorrectly}, NumberGuessedTotal: {numberOfTotalGuesses}, Accuracy: {((float) numberGuessedCorrectly / (float) numberOfTotalGuesses) * 100}%");
        }
    }

    private int FindHighestElement(List<double> collection)
    {
        //Debug.Log($"FHE collection: {collection.Count}");
        var prediction = collection[0];
        var numberPredicted = 0;
        for (var index = 1; index < collection.Count; index++)
        {
            if (prediction < collection[index])
            {
                prediction = collection[index];
                numberPredicted = index;
            }
        }
        return numberPredicted;
    }
    
    private void ParseDataFiles(string filePath, bool isTrainingData)
    {
        var streamReader = new StreamReader(filePath);
        string dataText;
        string[] currentInputString;
        if (isTrainingData)
        {
            while (!streamReader.EndOfStream)
            {
                var listOfTrainingValues = new List<double>();
                var listOfTrainingAnswers = new List<double>();
                var outputAnswersArray = new double[10];
                // read in the first line of the training data
                dataText = streamReader.ReadLine();
                // split the texts via a comma delimiter
                currentInputString = dataText?.Split(",");
                // trainingDataActual[counter] = new int[64, 1];
                // parse each character of the line 
                for (int i = 0; i < currentInputString?.Length - 1; i++)
                {
                    listOfTrainingValues.Add(double.Parse(currentInputString[i]));
                }
                listOfTrainingAnswers.Add(double.Parse(currentInputString[^1]));
                for (int i = 0; i < 10; i++)
                {
                    if (i == listOfTrainingAnswers[0])
                        outputAnswersArray[i] = 1;
                    else
                    {
                        outputAnswersArray[i] = 0;
                    }
                }
                setOfTrainingValues.Add(listOfTrainingValues);
                setOfTrainingAnswers.Add(outputAnswersArray.ToList());
            }
        }
        else
        {
            while (!streamReader.EndOfStream)
            {
                var listOfTestingValues = new List<double>();
                var listOfTestingAnswers = new List<double>();
                var outputAnswersArray = new double[10];
                // read in the first line of the training data
                dataText = streamReader.ReadLine();
                // split the texts via a comma delimiter
                currentInputString = dataText?.Split(",");
                // trainingDataActual[counter] = new int[64, 1];
                // parse each character of the line 
                for (int i = 0; i < currentInputString.Length - 1; i++)
                {
                    listOfTestingValues.Add(double.Parse(currentInputString[i]));
                }
                //listOfTestingAnswers.Add();
                for (int i = 0; i < 10; i++)
                {
                    if (i == double.Parse(currentInputString[^1]))
                        outputAnswersArray[i] = 1;
                    else
                    {
                        outputAnswersArray[i] = 0;
                    }
                }
                setOfTestingValues.Add(listOfTestingValues);
                setOfTestingAnswers.Add(outputAnswersArray.ToList());
            }
        }
    }
}
