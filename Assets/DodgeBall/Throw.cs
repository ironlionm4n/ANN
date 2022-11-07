using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Throw : MonoBehaviour
{
    public GameObject shperePrefab;
    public GameObject cubePrefab;
    public Material red;
    public Material green;
    private PerceptronDodge _perceptron;

    private void Start()
    {
        _perceptron = GetComponent<PerceptronDodge>();
    }

    private void Update()
    {
        if (Input.GetKeyDown("1"))
        {
            var g = Instantiate(shperePrefab, Camera.main.transform.position, Camera.main.transform.rotation);
            g.GetComponent<Renderer>().material = red;
            g.GetComponent<Rigidbody>().AddForce(0,0,500);
            _perceptron.FeedInput(0,0,0);
        }
        if (Input.GetKeyDown("2"))
        {
            var g = Instantiate(shperePrefab, Camera.main.transform.position, Camera.main.transform.rotation);
            g.GetComponent<Renderer>().material = green;
            g.GetComponent<Rigidbody>().AddForce(0,0,500);
            _perceptron.FeedInput(0,1,1);
        }
        if (Input.GetKeyDown("3"))
        {
            var g = Instantiate(cubePrefab, Camera.main.transform.position, Camera.main.transform.rotation);
            g.GetComponent<Renderer>().material = red;
            g.GetComponent<Rigidbody>().AddForce(0,0,500);
            _perceptron.FeedInput(1,0,1);
        }
        if (Input.GetKeyDown("4"))
        {
            var g = Instantiate(cubePrefab, Camera.main.transform.position, Camera.main.transform.rotation);
            g.GetComponent<Renderer>().material = green;
            g.GetComponent<Rigidbody>().AddForce(0,0,500);
            _perceptron.FeedInput(1,1,1);
        }
    }
}
