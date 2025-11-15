// static/js/capture.js
document.addEventListener('DOMContentLoaded', function() {
    const videoFeedCapture = document.getElementById('videoFeedCapture');
    const captureButton = document.getElementById('captureButton');
    const gestureLabelInput = document.getElementById('gestureLabel');
    const captureStatus = document.getElementById('captureStatus');

    let isCapturing = false;
    let frameCount = 0;
    const framesToCapture = 50; // Number of frames to capture for each gesture

    // Function to get a snapshot of the video feed as base64
    function getFrameAsBase64() {
        const img = new Image();
        img.src = videoFeedCapture.src; // Uses the current frame displayed by Flask
        const canvas = document.createElement('canvas');
        canvas.width = videoFeedCapture.naturalWidth;
        canvas.height = videoFeedCapture.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg');
    }

    captureButton.addEventListener('click', async () => {
        const gestureLabel = gestureLabelInput.value.trim();

        if (!gestureLabel) {
            alert("Please enter a gesture label (e.g., 'A', 'Hello').");
            return;
        }
        if (isCapturing) {
            alert("Already capturing. Please wait.");
            return;
        }

        isCapturing = true;
        captureButton.disabled = true;
        gestureLabelInput.disabled = true;
        captureStatus.textContent = `Capturing 0/${framesToCapture} frames for "${gestureLabel}"...`;
        frameCount = 0;

        const captureInterval = setInterval(async () => {
            if (frameCount < framesToCapture) {
                const imageDataUrl = getFrameAsBase64();
                try {
                    const response = await fetch('/save_gesture_data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            imageDataUrl: imageDataUrl,
                            gestureLabel: gestureLabel
                        })
                    });
                    const data = await response.json();
                    if (data.status === 'success') {
                        frameCount++;
                        captureStatus.textContent = `Capturing ${frameCount}/${framesToCapture} frames for "${gestureLabel}"...`;
                    } else {
                        console.error("Error saving frame:", data.message);
                        captureStatus.textContent = `Error: ${data.message} - Captured ${frameCount} frames.`;
                        clearInterval(captureInterval);
                        isCapturing = false;
                        captureButton.disabled = false;
                        gestureLabelInput.disabled = false;
                    }
                } catch (error) {
                    console.error("Network error:", error);
                    captureStatus.textContent = `Network Error: ${error.message} - Captured ${frameCount} frames.`;
                    clearInterval(captureInterval);
                    isCapturing = false;
                    captureButton.disabled = false;
                    gestureLabelInput.disabled = false;
                }
            } else {
                clearInterval(captureInterval);
                isCapturing = false;
                captureButton.disabled = false;
                gestureLabelInput.disabled = false;
                captureStatus.textContent = `Finished capturing ${framesToCapture} frames for "${gestureLabel}".`;
                console.log(`Finished capturing ${framesToCapture} frames for "${gestureLabel}".`);
            }
        }, 100); // Capture a frame every 100ms (10 frames per second)
    });
});