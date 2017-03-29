var fs = require('fs');
var xml2js = require('xml2js');
var BrainJSClassifier = require('natural-brain');
var classifier = new BrainJSClassifier();
var parser = new xml2js.Parser();

//Read the text from the file
fs.readFile(__dirname + '/data/reut2-000.sgm', function(err, data) {
	//Convert the XML to JSON
	parser.parseString(data, function (err, result) {
		//Get all of the entries
		var entries = result["DATA"]["REUTERS"];

		console.log("Adding " + entries.length + " entries...");

		//Add all entries to the neural network
		for(var i = 0; i < 100; i++){

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

		console.log("Getting classification...");
	    console.log(classifier.classify("Showers continued throughout the week"));
	});
});

function getInfoFromEntry(entry, callback){
	var topics = entry["TOPICS"][0]["D"] || undefined;
	var textObj = entry["TEXT"][0];
	var title = textObj["TITLE"];
	var body = textObj["BODY"];
		
	if(body !== undefined) body = body[0].replace(/(?:\r\n|\r|\n|\s+)/g, ' ');

	callback({"body": body, "topics": topics });
}