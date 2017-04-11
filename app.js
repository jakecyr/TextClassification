//Authors: Jake Cyr, Ryan Ek, Fanonx Rogers
var fs = require('fs'); //Read and write to files

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = null;
BrainJSClassifier.enableStopWords();

var nnFunctions = require("./neural_net_functions");
const saveFile = __dirname + "/classifier.json";

const filesToTrainOn = 18;

// Check if the save file exists
if(nnFunctions.fileExists(saveFile)) {

	console.log("Found save file...");
	console.log("Loading...");
	nnFunctions.load(saveFile, function(loadedClassifier){	
		console.log("Loaded brain...");
		classifier = loadedClassifier;


		console.log("Looking for all known labels...");
		nnFunctions.getCategories(saveFile, function(categories){
			console.log("Found all " + categories.length + " labels..");
			console.log("Starting to run tests...");
			nnFunctions.runAllTests(classifier, 0, categories, {}, function(newCounts){
				console.log("Ran all tests...");
				console.log("Saving test data...");

				console.log(newCounts);
				nnFunctions.saveValidationData(newCounts);
			});
		});
	});
}
else{
	classifier = new BrainJSClassifier();

	console.log("No saved file found...");
	console.log("Starting to train...");

	nnFunctions.train(0, filesToTrainOn, classifier, function(resultingClassifier){

		classifier = resultingClassifier;

		console.log(nnFunctions.getBrainWeights(classifier));

		nnFunctions.save(classifier, saveFile, function(){
			console.log("Saved to " + saveFile);
		});
	});
}