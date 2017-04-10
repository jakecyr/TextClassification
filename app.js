//Authors: Jake Cyr, Ryan Ek, Fanonx Rogers
var fs = require('fs'); //Read and write to files

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = new BrainJSClassifier();

var nnFunctions = require("./neural_net_functions");

//Location to save the classifier information learned
const saveFile = __dirname + "/classifier.json";
const baseFileName = __dirname + "/data/";
const filesToTrainOn = 18;

//Check if the save file exists
if (fileExists(saveFile)) {

	console.log("Found save file...");
	console.log("Loading...");

	nnFunctions.load(saveFile, function(loadedClassifier){

		// loadedClassifier.events.on('trainedWithDocument', function (obj) {console.log(obj); });
		
		console.log("Starting to train...");
		
		classifier = loadedClassifier;

		getCategories(function(categories){
			test(filesToTrainOn + 1, categories, {}, function(newCounts){
				var output = "CATEGORY,TP,TN,FP,FN,MCC\n";

				for(key in newCounts){
					var obj = newCounts[key];
					output += key + "," + obj.tp + "," + obj.tn + "," + obj.fp + "," + obj.fn + "," + calcMCC(obj.tp, obj.tn, obj.fp, obj.fn) + "\n";
				}

				fs.writeFile("testResults.csv", output, function(err){
					console.log("Test results saved to file");
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
		});
	});
}

function test(currentFile, categories, counts, callback){
	nnFunctions.test(classifier, categories, counts, baseFileName + currentFile + ".sgm", function(newCounts){
		if(currentFile == 21){
			callback(newCounts);
		}
		else{
			test(currentFile + 1, categories, newCounts, callback);
		}
	});
}
function calcMCC(tp, tn, fp, fn){
	var mcc = 0;
	var numerator = (tp * tn) - (fp * fn);
	var denominator = Math.sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) );
	if(denominator == 0) denominator = 1;
	return numerator/denominator;
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

		var output = [];

		for(key in cats){
			output.push(key);
		}

		callback(output);
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