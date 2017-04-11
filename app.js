//Authors: Jake Cyr, Ryan Ek, Fanonx Rogers
var fs = require('fs'); //Read and write to files

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = null;

var nnFunctions = require("./neural_net_functions");
const saveFile = __dirname + "/classifier.json";

// nnFunctions.kfold();

init()

function init(){
	// Check if the save file exists
	if(nnFunctions.fileExists(saveFile)) {
		nnFunctions.load(saveFile, function(loadedClassifier){	
			classifier = loadedClassifier;

			nnFunctions.getCategories(saveFile, function(categories){
				nnFunctions.runAllTests(classifier, 0, categories, {}, function(newCounts){
					console.log(newCounts);
					nnFunctions.saveValidationData(newCounts);
				});
			});
		});
	}
	else{
		classifier = new BrainJSClassifier();

		nnFunctions.train(0, classifier, function(resultingClassifier){
			classifier = resultingClassifier;
			nnFunctions.save(classifier, saveFile, function(){
				console.log("Saved to " + saveFile);
			});
		});
	}
}