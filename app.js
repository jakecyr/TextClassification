/*
 *Authors: Jake Cyr, Ryan Ek, Fanonx Rogers
 *
 */
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

const baseFileName = __dirname + "/data/";

var startTime = Date.now(); //Used for testing the training speeds

var testString = "argentine grain board figures show crop registrations of grains, oilseeds and their products to february 11, in thousands of tonnes";
//Start training the neural network classifier using the referenced file
var count = 0;

nnFunctions.load(saveFile, function(loadedClassifier){
	classifier = loadedClassifier;

	for(var i = 0; i < 22; i++){
		nnFunctions.train(classifier, baseFileName + i + ".sgm", function(result){
			if(count === 21){
				console.log(nnFunctions.classify(result, testString));
				console.log(nnFunctions.getClassifications(result, testString));
				console.log("It took " + ((Date.now() - startTime)/1000) + "s to train");
				nnFunctions.save(result, saveFile, function(){ console.log("Saved to " + saveFile);});
			}
			else{
				classifier = result;
			}

			count++;
		});
	}
});