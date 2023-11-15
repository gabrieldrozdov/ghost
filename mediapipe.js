import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

let faceLandmarker;
let runningMode = "VIDEO";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 1920;

// Before we can use HandLandmarker class we must wait for it to finish loading. Machine Learning models can be large and take a moment to get everything needed to run.
async function createFaceLandmarker() {
	const filesetResolver = await FilesetResolver.forVisionTasks(
		"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
	);
	faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
		baseOptions: {
			modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
			delegate: "GPU"
		},
		outputFaceBlendshapes: true,
		runningMode,
		numFaces: 20
	});
}
createFaceLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById(
	"output_canvas"
);

const canvasCtx = canvasElement.getContext("2d");

enableWebcamButton = document.getElementById("toggle");
enableWebcamButton.addEventListener("click", enableCam);

// Enable the live webcam view and start detection.
function enableCam(event) {
	let title = document.querySelector("h1");
	title.dataset.active = 0;
	setTimeout(() => {
		title.style.display = 'none';
	}, 2000)

	// getUsermedia parameters.
	const constraints = {
		video: true
	};

	if (!faceLandmarker) {
		console.log("Wait! faceLandmarker not loaded yet.");
		return;
	}

	enableWebcamButton.dataset.init = 0;
	if (webcamRunning === true) {
		webcamRunning = false;
		enableWebcamButton.dataset.active = 0;
	} else {
		webcamRunning = true;
		enableWebcamButton.dataset.active = 1;

		// Activate the webcam stream.
		navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
			video.srcObject = stream;
			video.addEventListener("loadeddata", predictWebcam);
		});
	}
}

let lastVideoTime = -1;
let results = undefined;
let features = {
	contours: false,
	righteye: true,
	righteyebrow: false,
	lefteye: true,
	lefteyebrow: false,
	face: false,
	lips: false,
	rightiris: true,
	leftiris: true,
}
const drawingUtils = new DrawingUtils(canvasCtx);
async function predictWebcam() {
	const radio = video.videoHeight / video.videoWidth;
	video.style.width = videoWidth + "px";
	video.style.height = videoWidth * radio + "px";
	canvasElement.width = video.videoWidth;
	canvasElement.height = video.videoHeight;
	// Now let's start detecting the stream.
	if (runningMode === "IMAGE") {
		runningMode = "VIDEO";
		await faceLandmarker.setOptions({ runningMode: runningMode });
	}
	let startTimeMs = performance.now();
	if (lastVideoTime !== video.currentTime) {
		lastVideoTime = video.currentTime;
		results = faceLandmarker.detectForVideo(video, startTimeMs);
	}
	if (results.faceLandmarks) {
		for (const landmarks of results.faceLandmarks) {
			if (features['contours']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_TESSELATION,
					{ color: "black", lineWidth: .2 }
				);
			}
			if (features['righteye']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
					{ color: "black", lineWidth: 2 }
				);
			}
			if (features['righteyebrow']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
					{ color: "black", lineWidth: 2 }
				);
			}
			if (features['lefteye']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
					{ color: "black", lineWidth: 2 }
				);
			}
			if (features['lefteyebrow']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
					{ color: "black", lineWidth: 2 }
				);
			}
			if (features['face']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
					{ color: "black", lineWidth: 2 }
				);
			}
			if (features['mouth']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_LIPS,
					{ color: "black", lineWidth: 2 }
				);
			}
			if (features['leftiris']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
					{ color: "black", lineWidth: 1 }
				);
			}
			if (features['rightiris']) {
				drawingUtils.drawConnectors(
					landmarks,
					FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
					{ color: "black", lineWidth: 1 }
				);
			}
		}
	}
	console.log(results.faceBlendshapes[0]);

	// Call this function again to keep predicting when the browser is ready.
	if (webcamRunning === true) {
		window.requestAnimationFrame(predictWebcam);
	}
}

for (let btn of document.querySelectorAll('[data-feature]')) {
	btn.addEventListener('click', () => {
		if (parseInt(btn.dataset.active) == 1) {
			features[btn.dataset.feature] = false;
			btn.dataset.active = 0;
		} else {
			features[btn.dataset.feature] = true;
			btn.dataset.active = 1;
		}
	})
}