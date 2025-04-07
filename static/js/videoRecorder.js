let mediaRecorder;
let recordedBlobs = [];
let stream;
let videoElement;
let recordButton;
let stopButton;
let playButton;
let submitButton;
let downloadButton;
let countdownElement;
let countdownInterval;

document.addEventListener("DOMContentLoaded", () => {
  videoElement = document.getElementById("webcam");
  recordButton = document.getElementById("recordButton");
  stopButton = document.getElementById("stopButton");
  playButton = document.getElementById("playButton");
  submitButton = document.getElementById("submitButton");
  downloadButton = document.getElementById("downloadButton");

  // Hide the stop, play, and download buttons as they're no longer needed
  if (stopButton) stopButton.style.display = "none";
  if (playButton) playButton.style.display = "none";
  if (downloadButton) downloadButton.style.display = "none";

  // Create countdown element
  countdownElement = document.createElement("div");
  countdownElement.id = "countdown";
  countdownElement.style.position = "absolute";
  countdownElement.style.top = "50%";
  countdownElement.style.left = "50%";
  countdownElement.style.transform = "translate(-50%, -50%)";
  countdownElement.style.fontSize = "6rem";
  countdownElement.style.fontWeight = "bold";
  countdownElement.style.color = "white";
  countdownElement.style.textShadow = "2px 2px 4px #000";
  countdownElement.style.zIndex = "100";
  countdownElement.style.display = "none";
  document.querySelector(".relative").appendChild(countdownElement);

  // Set up event listeners
  recordButton.addEventListener("click", startCountdown);
  // We'll keep these listeners in case the buttons are still in the DOM
  if (stopButton) stopButton.addEventListener("click", stopRecording);
  if (playButton) playButton.addEventListener("click", playRecording);
  if (downloadButton)
    downloadButton.addEventListener("click", downloadRecording);
  submitButton.addEventListener("click", submitRecording);

  // Initialize the webcam
  initWebcam();
});

// New function to start countdown
function startCountdown() {
  const countdownDuration = 10; // 10 seconds countdown
  let timeLeft = countdownDuration;

  // Disable record button during countdown
  recordButton.disabled = true;

  // Show and initialize the countdown display
  countdownElement.style.display = "block";
  countdownElement.textContent = timeLeft;

  // Start the countdown interval
  countdownInterval = setInterval(() => {
    timeLeft--;

    if (timeLeft > 0) {
      // Update countdown display
      countdownElement.textContent = timeLeft;
    } else {
      // Countdown finished
      clearInterval(countdownInterval);
      countdownElement.style.display = "none";

      // Start recording
      startRecording();
    }
  }, 1000);
}

async function initWebcam() {
  try {
    const constraints = {
      audio: false,
      video: {
        width: 480,
        height: 360,
      },
    };

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = stream;

    recordButton.disabled = false;
    console.log("getUserMedia() got stream:", stream);
  } catch (e) {
    console.error("navigator.getUserMedia error:", e);
    alert("Error accessing camera: " + e.message);
  }
}

function startRecording() {
  recordedBlobs = [];

  try {
    mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  } catch (e) {
    console.error("Exception while creating MediaRecorder:", e);
    alert("Error creating MediaRecorder: " + e.message);
    return;
  }

  console.log("Created MediaRecorder", mediaRecorder);
  recordButton.disabled = true;
  stopButton.disabled = false;
  playButton.disabled = true;
  downloadButton.disabled = true;
  submitButton.disabled = true;

  mediaRecorder.onstop = (event) => {
    console.log("Recorder stopped:", event);
  };

  mediaRecorder.ondataavailable = (event) => {
    console.log("Data available event:", event);
    if (event.data && event.data.size > 0) {
      recordedBlobs.push(event.data);
    }
  };

  // Start recording with a 3-second time slice
  mediaRecorder.start(3000);
  console.log("MediaRecorder started", mediaRecorder);

  // Set a timeout to automatically stop recording after 3 seconds
  setTimeout(() => {
    if (mediaRecorder.state === "recording") {
      stopRecording();
    }
  }, 3000);
}

function stopRecording() {
  if (countdownInterval) {
    clearInterval(countdownInterval);
    countdownElement.style.display = "none";
  }

  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    console.log("Recorded Blobs:", recordedBlobs);
  }

  recordButton.disabled = false;
  stopButton.disabled = true;
  playButton.disabled = false;
  downloadButton.disabled = false;
  submitButton.disabled = false;
}

function playRecording() {
  const recordedVideo = document.getElementById("recorded");
  const superBuffer = new Blob(recordedBlobs, { type: "video/webm" });
  recordedVideo.src = null;
  recordedVideo.srcObject = null;
  recordedVideo.src = window.URL.createObjectURL(superBuffer);
  recordedVideo.controls = true;
  recordedVideo.play();
}

function downloadRecording() {
  const blob = new Blob(recordedBlobs, { type: "video/webm" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = url;
  a.download = "workout.webm";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, 100);
}

function submitRecording() {
  const blob = new Blob(recordedBlobs, { type: "video/webm" });
  const formData = new FormData();
  formData.append("video", blob, "recorded-video.webm");

  // Get the form action URL from the data attribute
  const form = document.getElementById("recordForm");
  const url = form.getAttribute("data-action");

  // Show loading indicator
  document.getElementById("loadingIndicator").classList.remove("hidden");
  submitButton.disabled = true;

  fetch(url, {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (response.ok) {
        return response.text();
      }
      throw new Error("Network response was not ok.");
    })
    .then((html) => {
      // Replace the current page content with the result page
      document.open();
      document.write(html);
      document.close();
    })
    .catch((error) => {
      console.error("Error submitting video:", error);
      alert("Error submitting video: " + error.message);
      submitButton.disabled = false;
      document.getElementById("loadingIndicator").classList.add("hidden");
    });
}
