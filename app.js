var fs = require('fs'); //Read and write to files
var xml2js = require('xml2js'); //Convert XML to a JS object
var parser = new xml2js.Parser(); //Parse XML
var argv = require('yargs').argv; //To read arguments from the terminal

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = new BrainJSClassifier();

var nnFunctions = require("./neural_net_functions");

//Location to save the classifier information learned
const saveFile = __dirname + "/classifier.json";

var startTime = Date.now(); //Used for testing the training speeds

// Load the classifier data from the data file
nnFunctions.loadNeuralNetworkData("classifier.json", function(loadedClassifier){

	//Set the classifier to that of the one loaded from the file
	classifier = loadedClassifier;
	console.log("It took " + (Date.now() - startTime) + "ms to load");

	// Reset the start time for training
	startTime = Date.now();

	//Start training the neural network classifier using the referenced file
	nnFunctions.trainNeuralNetwork(classifier, __dirname + "/data/" + (argv.file || 1) + ".sgm", function(resultClassifier){
		//Update the classifier based on the recent training
		classifier = resultClassifier;

		console.log("It took " + ((Date.now() - startTime)/1000) + "s to train");
		
		//Save all of the training data to a file
		nnFunctions.saveNeuralNetwork(classifier, "classifier.json", function(result){
			console.log("Saved successfully");
		});
	});
});