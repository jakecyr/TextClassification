var fs = require('fs'); //Read and write to files
var xml2js = require('xml2js'); //Convert XML to a JS object
var parser = new xml2js.Parser(); //Parse XML

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = new BrainJSClassifier();

const wordsToTrain = 200; //Number of top words from each entry to train on
const filesToTrainOn = 18; //Number of files to use for training -- remaining files will be used for testing

const baseFileName = __dirname + "/data/"; //Base file name for all data files

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
		console.log("Loaded neural network data from file...");
		
		//Send the callback function the loaded classifier
		callback(classifier);
	});
}
//Save the current neural network state to a file
module.exports.save = function (classifierToSave, file, callback){
	classifierToSave.save(file, function(err, savedClassifier) {
		if(err) return console.log(err);
		console.log("Neural Network training data saved to file..."); 
		callback(savedClassifier);
	});
}
//Train the neural network using the specified file
module.exports.train = function (startFile, classifierToTrain, callback){
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
						if(data.topics[j] !== undefined){
							classifierToTrain.addDocument(data.body, data.topics[j]);
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
//Return an object given one of the entries from a data file
module.exports.getInfoFromEntry = function (entry){
	if(entry !== undefined){
		if(entry["TOPICS"] !== undefined){
			var topics = entry["TOPICS"][0]["D"] || undefined;
			var textObj = entry["TEXT"][0];
			var title = textObj["TITLE"];
			var body = textObj["BODY"];

			if(body !== undefined && topics !== undefined){ 
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
	}
}
//Performs k-fold cross validation on the data files
module.exports.kfold = function(){

	var index = parseInt(Math.random() * 22);
	var counts = {};
	var allData = [];

	readAllEntries(0, [], function(allEntries){

		const k = 10;

		//Split up all of the entries into 10 (mostly) equal groups
		var groups = splitUp(allEntries, k);

		//Test each group once
		for(var q = 0; q < k; q++){
			var classifier = new BrainJSClassifier(); //Create a new classifier

			var groupClone = groups.slice(0); //Clone the groups array

			var testing = groupClone.splice(q, 1); //remove the current testing group
			var training = groupClone; //set the remaining groups as training data

			//Loop through each training group
			for(var d = 0; d < training.length; d++){
				var currentTrainingSet = training[d];

				console.log(currentTrainingSet)

				//Loop through each entry in the current group
				for(var p = 0; p < currentTrainingSet.length; p++){
					var currentEntry = currentTrainingSet[p];

					//Get the important info from the entry
					var data = module.exports.getInfoFromEntry(currentEntry);
					
					//Make sure data was found (might not be since some entries have inconsistencies in XML)
					if(data != undefined){
						//Make sure there are both topics and a body for the current datum
						if(data.topics !== undefined && data.body !== undefined){
							//Loop through all of the topics
							for(var j = 0; j < data.topics.length; j++){
								classifier.addDocument(data.body, data.topics[j]); //Add the topic/body to the classifier
							}
						}
					}

				}

				console.log("Starting to train");
				classifier.train(); //train the classifier with data from the current training set
			}


			//Loop through each entry in the testing set
			for(var i = 0; i < testing.length; i++){
				var currentTestingSet = training[i];

				for(var p = 0; p < currentTrainingSet.length; p++){
					var currentEntry = currentTrainingSet[p];

					//Get the important info from the entry
					var data = module.exports.getInfoFromEntry(currentEntry);
					
					//Make sure data was found (might not be since some entries have inconsistencies in XML)
					if(data != undefined){
						//Make sure there are both topics and a body for the current datum
						if(data.topics !== undefined && data.body !== undefined){
							//Loop through all of the topics
							for(var j = 0; j < data.topics.length; j++){
								classifier.addDocument(data.body, data.topics[j]); //Add the topic/body to the classifier
							}
						}
					}
				}
			}


			//Test Entries with teh neural network
			for(var i = 0; i < entries.length; i++){

				var currentEntry = entries[i];
				var data = module.exports.getInfoFromEntry(currentEntry);

				if(data.topics !== undefined && data.body !== undefined){
					var classificationObj = module.exports.getClassifications(classifierToTest, data.body);

					var classifications = [];

					for(var o = 0; o < classificationObj.length; o++){
						if(classificationObj[o].value > 0.2){
							classifications.push(classificationObj[o].label);
						}
					}

					if(classifications.length == 0) classifications.push(module.exports.classify(classifierToTest, data.body));

					//Loop through each known category
					for(var l = 0; l < categories.length; l++){
						var currentCategory = categories[l];

						//Check if the current category is in the array of topics or classifications
						var catInTopics = (data.topics.indexOf(currentCategory) > -1) ? true : false;
						var catInClassification = (classifications.indexOf(currentCategory)) > -1 ? true : false;

						// console.log("Category", currentCategory, "TOPICS", data.topics, "CLASS", classifications);

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

		}
	});
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

					for(var o = 0; o < classificationObj.length; o++){
						if(classificationObj[o].value > 0.2){
							classifications.push(classificationObj[o].label);
						}
					}

					if(classifications.length == 0) classifications.push(module.exports.classify(classifierToTest, data.body));

					//Loop through each known category
					for(var k = 0; k < categories.length; k++){
						var currentCategory = categories[k];

						//Check if the current category is in the array of topics or classifications
						var catInTopics = (data.topics.indexOf(currentCategory) > -1) ? true : false;
						var catInClassification = (classifications.indexOf(currentCategory)) > -1 ? true : false;

						// console.log("Category", currentCategory, "TOPICS", data.topics, "CLASS", classifications);

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
		if(currentFile == 21)
			callback(newCounts);
		else
			module.exports.runAllTests(classifier, currentFile + 1, categories, newCounts, callback);
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
//Calculates 
module.exports.mcc = function(tp, tn, fp, fn){
	var mcc = 0;
	var numerator = (tp * tn) - (fp * fn);
	var denominator = Math.sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) );
	if(denominator == 0) denominator = 1;
	return numerator/denominator;
}
//Returns an array of categories (topics) from the data files
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
//Checks if a referenced file exists
module.exports.fileExists = function(filePath){
	try{
		return fs.statSync(filePath).isFile();
	}
	catch (err){
		return false;
	}
}

//Returns an array that is randomly shuffled
function shuffle(array) {
	var currentIndex = array.length, temporaryValue, randomIndex;

	// While there remain elements to shuffle...
	while (0 !== currentIndex) {

		// Pick a remaining element...
		randomIndex = Math.floor(Math.random() * currentIndex);
		currentIndex -= 1;

		// And swap it with the current element.
		temporaryValue = array[currentIndex];
		array[currentIndex] = array[randomIndex];
		array[randomIndex] = temporaryValue;
	}

	return array;
}
//Splits an array in n arrays with (mostly) equal number of elements
function splitUp(arr, n) {
    var rest = arr.length % n, // how much to divide
        restUsed = rest, // to keep track of the division over the elements
        partLength = Math.floor(arr.length / n),
        result = [];

        for(var i = 0; i < arr.length; i += partLength) {
        	var end = partLength + i,
        	add = false;

        if(rest !== 0 && restUsed) { // should add one element for the division
        	end++;
            restUsed--; // we've used one division element now
            add = true;
        }

        result.push(arr.slice(i, end)); // part of the array

        if(add) {
            i++; // also increment i in the case we added an extra element for division
        }
    }

    return result;
}
//Reads and returns all of the entries from the data files
function readAllEntries(file, entries, callback){
	fs.readFile(baseFileName + file + ".sgm", function(err, data) {
		//Convert the XML to JSON
		parser.parseString(data, function (err, result) {
			
			//Get all of the entries
			var entries = result["DATA"]["REUTERS"];

			for(var i = 0; i < entries.length; i++){

				var currentEntry = entries[i];

				if(currentEntry !== undefined){

					//Get the important info from the entry
					var data = module.exports.getInfoFromEntry(currentEntry);
					
					if(data !== undefined){
						if(data.topics !== undefined && data.body !== undefined){
							for(var j = 0; j < data.topics.length; j++){
								if(data.topics[i] !== undefined){
									entries.push(data);
								}
							}
						}
					}
				}
			}

			if(file == 21){
				callback(entries);
			}
			else{
				readAllEntries(file + 1, entries, callback);
			}
		});
	});
}