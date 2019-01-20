let line;
let square;
let triangle;

const total_data = 1000;
const dim = 28;
const train_num = 800;

let line_data = {
	label: 0
};
let square_data = {
	label: 1
};
let triangle_data = {
	label: 2
};

let nn;

let userTest = [];

function trainingEpoch() {
	let training_data = [];
	training_data = training_data.concat(line_data.training);
	training_data = training_data.concat(square_data.training);
	training_data = training_data.concat(triangle_data.training);
	shuffle(training_data, true);

	for (let i = 0; i < training_data.length; i++) {
		let data = [];
		for (let j = 0; j < training_data[i].length; j++) {
			data[j] = [training_data[i][j] / 255.0];
		}

		let target = [
			[0],
			[0],
			[0]
		];
		target[training_data[i].label] = [1];

		nn.train(data, target);
	}
}

function testAll() {
	let testing_data = [];
	testing_data = testing_data.concat(line_data.testing);
	testing_data = testing_data.concat(square_data.testing);
	testing_data = testing_data.concat(triangle_data.testing);

	let correct = 0;

	for (let i = 0; i < testing_data.length; i++) {
		let data = [];
		for (let j = 0; j < testing_data[i].length; j++) {
			data[j] = [testing_data[i][j] / 255.0];
		}

		let output = nn.predict(data);
		let classification = [];
		classification[0] = output[0][0];
		classification[1] = output[1][0];
		classification[2] = output[2][0];
		let guess = classification.indexOf(max(classification));

		if (guess === testing_data[i].label) {
			correct++;
		}
	}

	let percent = correct / testing_data.length;
	return percent;
}

function prepareData(category, data) {
	category.training = [];
	category.testing = [];
	let limit = floor(0.8 * total_data);
	for (var i = 0; i < total_data; i++) {
		let offset = i * dim * dim;
		if (i < limit) {
			category.training[i] = data.bytes.subarray(offset, offset + dim * dim);
			category.training[i].label = category.label;
		} else {
			category.testing[i - limit] = data.bytes.subarray(offset, offset + dim * dim);
			category.testing[i - limit].label = category.label;
		}
	}
}

function preload() {
	line = loadBytes('./outData/line1000.bin');
	square = loadBytes('./outData/square1000.bin');
	triangle = loadBytes('./outData/triangle1000.bin');
}

function setup() {
	createCanvas(280, 280);

	prepareData(line_data, line);
	prepareData(square_data, square);
	prepareData(triangle_data, triangle);

	nn = new NeuralNetwork(784, 64, 3, 0.1);

	console.log("Without training: ", testAll() * 100);
	for (let i = 1; i < 6; i++) {
		console.log("TRAINING FOR ", i, " EPOCH");
		trainingEpoch();
		console.log("PERCENT AFTER: ", testAll() * 100);
	}
	console.log("FINALLY: ", testAll() * 100);
}

function draw() {
	background(0);
	strokeWeight(16);
	stroke(255);
	noFill();
	beginShape();
	for (let i = 0; i < userTest.length; i++) {
		vertex(userTest[i][0], userTest[i][1]);
	}
	endShape();
}

function mouseReleased() {
	userTest.push([mouseX, mouseY]);
}

function keyPressed() {
	if (keyCode === UP_ARROW) {
		background(0);
		userTest = [];
	} else {
		let input = [];
		let img = get();
		img.resize(28, 28);
		img.loadPixels();
		for (let i = 0; i < dim * dim; i++) {
			let brightness = img.pixels[i * 4];
			input[i] = [brightness / 255.0];
		}
		let output = nn.predict(input);
		let classification = [];
		classification[0] = output[0][0];
		classification[1] = output[1][0];
		classification[2] = output[2][0];
		let guess = classification.indexOf(max(classification));
		if (guess === 0) {
			console.log("LINE");
		} else if (guess === 1) {
			console.log("SQUARE");
		} else if (guess === 2) {
			console.log("TRIANGLE");
		}
	}
}