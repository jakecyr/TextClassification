var fs = require('fs');
var xml2js = require('xml2js');
var BrainJSClassifier = require('natural-brain');

var classifier = new BrainJSClassifier();
var parser = new xml2js.Parser();

const saveFile = __dirname + "/classifier.json";

loadNeuralNetworkData("classifier.json", function(loadedClassifier){
	classifier = loadedClassifier;

	trainNeuralNetwork(__dirname + "/data/reut2-000.sgm", function(){
		saveNeuralNetwork("classifier.json", function(result){
			console.log("Saved successfully");
		});
	});
})

//Load the neural network data from the save file
function loadNeuralNetworkData(file, callback){
	BrainJSClassifier.load(file, null, function(err, classifier) {
		if(err) return console.log(err);
		console.log("Loaded data from file...");
		callback(classifier);
	});
}
//Train the neural network using the specified file
function trainNeuralNetwork(trainingTextFilename, callback){
	//Read text from the file to train the neural network
	fs.readFile(trainingTextFilename, function(err, data) {
		
		//Convert the XML to JSON
		parser.parseString(data, function (err, result) {
			
			//Get all of the entries
			var entries = result["DATA"]["REUTERS"];

			console.log("Adding " + entries.length + " entries...");

			//Add all entries to the neural network
			for(var i = 0; i < 50; i++){

				var currentEntry = entries[i];

				//Get the important info from the entry
				getInfoFromEntry(currentEntry, function(data){
					if(data.topics !== undefined && data.body !== undefined){
						classifier.addDocument(data.body, data.topics[0]);
					}
				});
			}

			console.log("Starting to train...");
			classifier.train();
			console.log("Done training.");

			callback();
		});
	});
}
//Save the current neural network state to a file
function saveNeuralNetwork(file, callback){
	classifier.save(file, function(err, classifier) {
		if(err) return console.log(err);
		console.log("Neural Network training data saved to " + saveFile); 
		callback(classifier);
	});
}
//Return an object given one of the entries from a data file
function getInfoFromEntry(entry, callback){
	var topics = entry["TOPICS"][0]["D"] || undefined;
	var textObj = entry["TEXT"][0];
	var title = textObj["TITLE"];
	var body = textObj["BODY"];

	if(body !== undefined) body = body[0].replace(/(?:\r\n|\r|\n|\s+|a|the|and)/g, ' ');

	callback({"body": body, "topics": topics });
}