var fs = require('fs'); //Read and write to files
var xml2js = require('xml2js'); //Convert XML to a JS object
var parser = new xml2js.Parser(); //Parse XML
var argv = require('yargs').argv; //To read arguments from the terminal

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = new BrainJSClassifier();

const wordsToTrain = 100; //Number of top words from each entry to train on

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
module.exports.train = function (classifierToTrain, trainingTextFilename, callback){
	//Read text from the file to train the neural network

	fs.readFile(trainingTextFilename, function(err, data) {
		
		//Convert the XML to JSON
		parser.parseString(data, function (err, result) {
			
			//Get all of the entries
			var entries = result["DATA"]["REUTERS"];

			console.log("Adding " + entries.length + " entries...");

			//Add all entries to the neural network
			for(var i = 0; i < entries.length; i++){

				var currentEntry = entries[i];

				//Get the important info from the entry
				module.exports.getInfoFromEntry(currentEntry, function(data){
					if(data.topics !== undefined && data.body !== undefined){
						for(var j = 0; j < data.topics.length; j++){
							if(data.topics[i] !== undefined) if(data.body.length > 0) classifierToTrain.addDocument(data.body, data.topics[i]);
						}
					}
				});
			}

			console.log("Starting to train...");

			//Train the classifier with the added documents
			classifierToTrain.train();

			console.log("Done training.");

			//Return the classifier trained with the new information
			callback(classifierToTrain);
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
module.exports.getInfoFromEntry = function (entry, callback){
	var topics = entry["TOPICS"][0]["D"] || undefined;
	var textObj = entry["TEXT"][0];
	var title = textObj["TITLE"];
	var body = textObj["BODY"];

	if(body !== undefined){ 
		body = body[0].replace(/(?:\r\n|\r|\n|\s+|\sa\s|\sthe\s|\sand\s|\.|\')/g, ' ');

		var words = body.split(" ");
		var wordCounts = {};

		for(var p = 0; p < words.length; p++){
			var word = words[p].trim().toUpperCase();

			if(word != "" && /^[a-zA-Z0-9]+$/i.test(word) && word != "THE" && word != "A" && word != "AND" && word != "REUTER"){
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
	}

	callback({"body": body, "topics": topics });
}