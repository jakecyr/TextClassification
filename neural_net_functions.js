var fs = require('fs'); //Read and write to files
var xml2js = require('xml2js'); //Convert XML to a JS object
var parser = new xml2js.Parser(); //Parse XML

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = new BrainJSClassifier();

const wordsToTrain = 200; //Number of top words from each entry to train on
const filesToTrainOn = 18;

const baseFileName = __dirname + "/data/";

//Return a classification for a referenced string using a trained classifier
module.exports.classify = function (classifierToUse, stringToClassify){
	return classifierToUse.classify(stringToClassify);
}
//Returns all of the labels faced so far with their likelihood of 
//matching the referenced string
module.exports.getClassifications = function (classifierToUse, stringToClassify){
	return classifierToUse.getClassifications(stringToClassify);
}
//Load the neural network data from the save file
module.exports.load = function (file, callback){
	//Load the classifier from the file
	BrainJSClassifier.load(file, null, function(err, classifier) {
		if(err) return console.log(err);
		console.log("Loaded data from file " + file + "...");
		
		//Send the callback function the loaded classifier
		callback(classifier);
	});
}
//Train the neural network using the specified file
module.exports.train = function (startFile, filesToTrainOn, classifierToTrain, callback){
	//Read text from the file to train the neural network
	fs.readFile(baseFileName + startFile + ".sgm", function(err, data) {
		//Convert the XML to JSON
		parser.parseString(data, function (err, result) {
			
			//Get all of the entries
			var entries = result["DATA"]["REUTERS"];

			//Add all entries to the neural network
			for(var i = 0; i < entries.length; i++){

				var currentEntry = entries[i];

				//Get the important info from the entry
				var data = module.exports.getInfoFromEntry(currentEntry);
				
				if(data.topics !== undefined && data.body !== undefined){
					for(var j = 0; j < data.topics.length; j++){
						if(data.topics[i] !== undefined){
							classifierToTrain.addDocument(data.body, data.topics[i]);
						}
					}
				}
			}

			if(startFile == filesToTrainOn){
				console.log("TRAINING");
				classifierToTrain.train();
				callback(classifierToTrain);
			}
			else{
				startFile += 1;
				module.exports.train(startFile, filesToTrainOn, classifierToTrain, callback);
			}
		});
	});
}
//Save the current neural network state to a file
module.exports.save = function (classifierToSave, file, callback){
	classifierToSave.save(file, function(err, savedClassifier) {
		if(err) return console.log(err);
		console.log("Neural Network training data saved to " + file); 
		callback(savedClassifier);
	});
}
//Return an object given one of the entries from a data file
module.exports.getInfoFromEntry = function (entry){
	var topics = entry["TOPICS"][0]["D"] || undefined;
	var textObj = entry["TEXT"][0];
	var title = textObj["TITLE"];
	var body = textObj["BODY"];

	if(body !== undefined){ 
		body = body[0].replace(/(?:\r\n|\r|\n|\s+|\.|\')/g, ' ');

		var words = body.split(" ");
		var wordCounts = {};

		for(var p = 0; p < words.length; p++){
			var word = words[p].trim();

			if(word != "" && /^[a-zA-Z0-9]+$/i.test(word) && word.length > 4){
				if(word in wordCounts){
					wordCounts[word] = wordCounts[word] + 1;
				}
				else{
					wordCounts[word] = 0;
				}
			}
		}

		var sortable = [];

		for (var word in wordCounts) {
			sortable.push([word, wordCounts[word]]);
		}

		var sortedValues = sortable.sort(function(a, b) {
			return a[1] - b[1];
		});

		body = [];

		//Only add the top 5 words
		for(var p = 0; p < sortedValues.length && p < wordsToTrain; p++){
			body.push(sortedValues[p][0]);
		}
		
		return {"body": body, "topics": topics };
	}
	else{
		return {body: undefined, topics: undefined};
	}
}
//Test the neural network using the specified file
module.exports.test = function (classifierToTest, categories, counts, trainingTextFilename, callback){
	//Read text from the file to train the neural network

	fs.readFile(trainingTextFilename, function(err, data) {
		
		//Convert the XML to JSON
		parser.parseString(data, function (err, result) {
			
			//Get all of the entries
			var entries = result["DATA"]["REUTERS"];

			//Test Entries with teh neural network
			for(var i = 0; i < entries.length; i++){

				var currentEntry = entries[i];
				var data = module.exports.getInfoFromEntry(currentEntry);

				if(data.topics !== undefined && data.body !== undefined){
					var classificationObj = module.exports.getClassifications(classifierToTest, data.body);

					var classifications = [];

					for(var o = 0; o < classificationObj.length; o++) classifications.push(classificationObj[o].label);

					//Loop through each known category
					for(var k = 0; k < categories.length; k++){
						var currentCategory = categories[k];

						//Check if the current category is in the array of topics or classifications
						var catInTopics = (data.topics.indexOf(currentCategory) > -1) ? true : false;
						var catInClassification = (classifications.indexOf(currentCategory)) > -1 ? true : false;

						// console.log(currentCategory, data.topics, classifications);

						//True positive
						if(catInTopics && catInClassification){
							if(counts[currentCategory]){
								counts[currentCategory]["fp"] = counts[currentCategory]["fp"]+1;
							}
							else{
								counts[currentCategory] = {"tp": 1, "tn": 0, "fp": 0, "fn": 0 };
							}
						}
						//True negative
						else if(!catInTopics && !catInClassification){
							if(counts[currentCategory]){
								counts[currentCategory]["tn"] = counts[currentCategory]["tn"]+1;
							}
							else{
								counts[currentCategory] = {"tp": 0, "tn": 1, "fp": 0, "fn": 0 };
							}
						}
						//False positive
						else if(!catInTopics && catInClassification){
							if(counts[currentCategory]){
								counts[currentCategory]["fp"] = counts[currentCategory]["fp"]+1;
							}
							else{
								counts[currentCategory] = {"tp": 0, "tn": 0, "fp": 1, "fn": 0 };
							}
						}
						//False negative
						else if(catInTopics && !catInClassification){
							if(counts[currentCategory]){
								counts[currentCategory]["fn"] = counts[currentCategory]["fn"]+1;
							}
							else{
								counts[currentCategory] = {"tp": 0, "tn": 0, "fp": 0, "fn": 1 };
							}
						}
					}
				}
			}

			//Return the classifier trained with the new information
			callback(counts);
		});
	});
}
//Returns an array with the weights associated with each connection
module.exports.getBrainWeights = function(classifierWithWeights){
	return classifierWithWeights.classifier.brain.weights;
}
//Run tests with all of the files that weren't trained on
module.exports.runAllTests = function(classifier, currentFile, categories, counts, callback){
	module.exports.test(classifier, categories, counts, baseFileName + currentFile + ".sgm", function(newCounts){
		if(currentFile == 21){
			callback(newCounts);
		}
		else{
			// console.log(newCounts);
			module.exports.runAllTests(classifier, currentFile + 1, categories, newCounts, callback);
		}
	});
}
module.exports.saveValidationData = function(counts){
	var output = "CATEGORY,TP,TN,FP,FN,MCC\n";

	for(key in counts){
		var obj = counts[key];
		output += key + "," + obj.tp + "," + obj.tn + "," + obj.fp + "," + obj.fn + "," + module.exports.mcc(obj.tp, obj.tn, obj.fp, obj.fn) + "\n";
	}

	fs.writeFile("testResults.csv", output, function(err){
		console.log("Test results saved to file");
	});
}
module.exports.mcc = function(tp, tn, fp, fn){
	var mcc = 0;
	var numerator = (tp * tn) - (fp * fn);
	var denominator = Math.sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) );
	if(denominator == 0) denominator = 1;
	return numerator/denominator;
}
module.exports.getCategories = function(saveFile, callback){
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
module.exports.fileExists = function(filePath){
	try{
		return fs.statSync(filePath).isFile();
	}
	catch (err){
		return false;
	}
}