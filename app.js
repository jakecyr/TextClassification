//Authors: Jake Cyr, Ryan Ek, Fanonx Rogers
var fs = require('fs'); //Read and write to files

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = new BrainJSClassifier();

var nnFunctions = require("./neural_net_functions");

//Location to save the classifier information learned
const saveFile = __dirname + "/classifier.json";
const baseFileName = __dirname + "/data/";
const filesToTrainOn = 17;

//Check if the save file exists
if (fileExists(saveFile)) {

	console.log("Found save file...");
	console.log("Loading...");

	nnFunctions.load(saveFile, function(loadedClassifier){
		
		console.log("Starting to train...");
		
		classifier = loadedClassifier;

		trainOnFiles(filesToTrainOn, function(){
			nnFunctions.save(classifier, saveFile, function(){
				console.log("Saved to " + saveFile);

				getCategories(function(categories){
					nnFunctions.test(classifier, categories,  baseFileName + "0.sgm", function(resultingCounts){
						console.log(resultingCounts);
					});
				});
			});
		});
	});
}
else{
	console.log("No saved file found...");
	console.log("Starting to train...");

	trainOnFiles(filesToTrainOn, function(){
		nnFunctions.save(classifier, saveFile, function(){
			console.log("Saved to " + saveFile);

			getCategories(function(categories){
				nnFunctions.test(classifier, categories,  baseFileName + "0.sgm", function(resultingCounts){
					console.log(resultingCounts);
				});
			});
		});
	});
}

function trainOnFiles(maxFileToTrainOn, callback){
	var count = 0;

	for(var i = 0; i < maxFileToTrainOn; i++){
		nnFunctions.train(classifier, baseFileName + i + ".sgm", function(result){
			if(count === maxFileToTrainOn - 1){
				return callback();
			}
			else{
				classifier = result;
			}

			count++;
		});
	}
}

function getCategories(callback){
	fs.readFile(saveFile, function(err, jsonResult){
		if(err) return console.error(err);

		var json = JSON.parse(jsonResult);
		var labels = json.docs;
		var cats = {};

		for(var j = 0; j < labels.length; j++){
			cats[labels[j].label] = 1;
		}

		callback(cats);
	});
}

function fileExists(filePath){
    try {
        return fs.statSync(filePath).isFile();
    }
    catch (err) {
        return false;
    }
}