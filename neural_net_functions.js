var fs = require('fs'); //Read and write to files
var xml2js = require('xml2js'); //Convert XML to a JS object
var parser = new xml2js.Parser(); //Parse XML

//Neural Network and NLP imports
var BrainJSClassifier = require('natural-brain'); 
var classifier = new BrainJSClassifier();

const wordsToTrain = 200; //Number of top words from each entry to train on
const baseFileName = __dirname + "/data/"; //Base file name for all data files

//Performs k-fold cross validation on the data files
module.exports.kfold = function(){

	var counts = {};

	//Get an array of the entries from all of the data files
	readAllEntries(0, [], function(allEntries){

		const k = 10;

		allEntries = shuffle(allEntries);

		//Split up all of the entries into 10 (mostly) equal groups
		var groups = splitUp(allEntries, k);

		//Test each group once
		for(var q = 0; q < k; q++){
			var classifier = new BrainJSClassifier(); //Create a new classifier

			var testing = [];
			var training = [];

			testing = groups.slice(0).splice(q, 1);

			for(var a = 0; a < groups.length; a++) if(a != q) training.push(allEntries[a]);

			//Loop through each training group
			for(var d = 0; d < training.length; d++){
				var currentTrainingSet = training[d];

				//Loop through all of the topics
				for(var j = 0; j < currentTrainingSet.topics.length; j++){
					classifier.addDocument(currentTrainingSet.body, currentTrainingSet.topics[j]); //Add the topic/body to the classifier
				}

				classifier.train(); //train the classifier with data from the current training set
				console.log(`Finished training set ${d}...`);
			}

			console.log("Finished training...");

			//Get all categories from the classifier
			var categories = getCategories(classifier);

			console.log("Starting to test...");

			testing = testing[0];

			//Loop through each entry in the testing set
			for(var i = 0; i < testing.length; i++){

				var currentTestingSet = testing[i];

				//Make sure there are both topics and a body for the current datum
				if(currentTestingSet.topics !== undefined && currentTestingSet.body !== undefined){
					
					var classificationObj = getClassifications(classifier, currentTestingSet.body);
					var classifications = [];

					for(var o = 0; o < classificationObj.length; o++) classifications.push(classificationObj[o].label);
						
					if(classifications.length == 0) classifications.push(classify(classifier, currentTestingSet.body));

					//Loop through each known category
					for(var l = 0; l < categories.length; l++){
						var currentCategory = categories[l];

						//Check if the current category is in the array of topics or classifications
						var catInTopics = (currentTestingSet.topics.indexOf(currentCategory) > -1) ? true : false;
						var catInClassification = (classifications.indexOf(currentCategory)) > -1 ? true : false;

						// console.log("Category", currentCategory, "TOPICS", data.topics, "CLASS", classifications);

						//True positive
						if(catInTopics && catInClassification){
							if(counts[currentCategory]){
								counts[currentCategory]["tp"] = counts[currentCategory]["tp"]+1;
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

		//Save the confusion matrix counts with the corresponding mcc value to a file
		saveValidationData(counts);
	});
}

//Return a classification for a referenced string using a trained classifier
function classify(classifierToUse, stringToClassify){
	return classifierToUse.classify(stringToClassify);
}
//Returns all of the labels faced so far with their likelihood of 
//matching the referenced string
function getClassifications(classifierToUse, stringToClassify){
	return classifierToUse.getClassifications(stringToClassify);
}
//Return an object given one of the entries from a data file
function getInfoFromEntry(entry){
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
function saveValidationData(counts){
	var output = "CATEGORY,TP,TN,FP,FN,MCC\n";

	for(key in counts){
		var obj = counts[key];
		output += key + "," + obj.tp + "," + obj.tn + "," + obj.fp + "," + obj.fn + "," + mcc(obj.tp, obj.tn, obj.fp, obj.fn) + "\n";
	}

	fs.writeFile("testResults.csv", output, function(err){
		console.log("Test results saved to file");
	});
}
//Returns an array of categories (topics) from the data files
function getCategories(classifier){
	var labels = classifier.docs;
	var cats = {};
	var output = [];

	for(var j = 0; j < labels.length; j++) cats[labels[j].label] = 1;
	for(key in cats) output.push(key);

	return output;
}
//Calculates 
function mcc(tp, tn, fp, fn){
	var mcc = 0;
	var numerator = (tp * tn) - (fp * fn);
	var denominator = Math.sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) );
	if(denominator == 0) denominator = 1;
	return numerator/denominator;
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

        if(add) i++;
    }

    return result;
}
//Reads and returns all of the entries from the data files
function readAllEntries(file, entries, callback){
	fs.readFile(baseFileName + file + ".sgm", function(err, data) {
		//Convert the XML to JSON
		parser.parseString(data, function (err, result) {
			
			//Get all of the entries
			var newEntries = result["DATA"]["REUTERS"];

			for(var i = 0; i < newEntries.length; i++){

				var currentEntry = newEntries[i];

				if(currentEntry !== undefined){

					//Get the important info from the entry
					var data = getInfoFromEntry(currentEntry);
					
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