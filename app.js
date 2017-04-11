//Authors: Jake Cyr, Ryan Ek, Fanonx Rogers
var fs = require('fs'); //Read and write to files

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = null;

var nnFunctions = require("./neural_net_functions");
const saveFile = __dirname + "/classifier.json";

nnFunctions.kfold();

function testFunctions(){
	// Check if the save file exists
	if(nnFunctions.fileExists(saveFile)) {

		//Load the classifier from the save file
		nnFunctions.load(saveFile, function(loadedClassifier){	
			classifier = loadedClassifier; //Store the loaded classifier

			var counts = {};
			var categories = nnFunctions.getCategories(classifier);

			categories.forEach(function(data){
				counts[data] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0 };
			})

			//Runs tests on all docs (for testing purposes)
			nnFunctions.runAllTests(classifier, 0, categories, counts, function(newCounts){
				//Save the confusion matrix data to the testResults.csv file
				nnFunctions.saveValidationData(newCounts);
			});
		});
	}
	//No save file found
	else{
		classifier = new BrainJSClassifier(); //Create a new classifier

		//Train the classifier
		nnFunctions.train(0, classifier, function(resultingClassifier){
			classifier = resultingClassifier; //Update the classifier
			
			//Save the trained classifier
			nnFunctions.save(classifier, saveFile, function(){
				console.log("Saved to " + saveFile);
			});
		});
	}
}