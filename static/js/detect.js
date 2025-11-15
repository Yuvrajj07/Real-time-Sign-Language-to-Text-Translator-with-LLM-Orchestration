document.addEventListener('DOMContentLoaded', function() {
    const videoFeedDetect = document.getElementById('videoFeedDetect');
    const detectionStatus = document.getElementById('detectionStatus');
    const saveWordButton = document.getElementById('saveWordButton');
    const deleteLastWordButton = document.getElementById('deleteLastWordButton');
    const generateSentenceButton = document.getElementById('generateSentenceButton'); // NEW ID
    const wordsRecognizedDisplay = document.getElementById('wordsRecognizedDisplay'); 
    const savedSentenceDisplay = document.getElementById('savedSentenceDisplay');

    videoFeedDetect.onload = () => {
        detectionStatus.textContent = "Detecting gestures in real-time...";
    };

    videoFeedDetect.onerror = () => {
        detectionStatus.textContent = "Error loading video feed. Make sure your webcam is available and the server is running.";
    };
    
    // Helper function to handle button response and display updates
    function updateRecognizedWordsDisplay(data, statusMessage) {
        // Only update the Words Recognized list when saving/deleting
        wordsRecognizedDisplay.textContent = data.current_sentence || " ";
        // savedSentenceDisplay is intentionally NOT updated here
        detectionStatus.textContent = statusMessage;
    }

    saveWordButton.addEventListener('click', async () => {
        saveWordButton.disabled = true; 
        detectionStatus.textContent = "Attempting to save word...";

        try {
            const response = await fetch('/save_word', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}) 
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                updateRecognizedWordsDisplay(data, `Saved: "${data.last_word_saved}" | Detection running...`);
            } else {
                detectionStatus.textContent = `Error saving word: ${data.message}`;
            }

        } catch (error) {
            console.error("Network error saving word:", error);
            detectionStatus.textContent = `Network Error while saving word: ${error.message}`;
        } finally {
            setTimeout(() => { saveWordButton.disabled = false; }, 500); 
        }
    });

    deleteLastWordButton.addEventListener('click', async () => {
        deleteLastWordButton.disabled = true;
        detectionStatus.textContent = "Attempting to delete last word...";

        try {
            const response = await fetch('/delete_last_word', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}) 
            });

            const data = await response.json();

            if (data.status === 'success') {
                const message = data.deleted_word ? `Deleted: "${data.deleted_word}" | Detection running...` : "Sentence cleared. Detection running...";
                // Use the helper function to update the recognized words list
                updateRecognizedWordsDisplay(data, message); 
            } else {
                detectionStatus.textContent = `Error deleting word: ${data.message}`;
            }
        } catch (error) {
            console.error("Network error deleting word:", error);
            detectionStatus.textContent = `Network Error while deleting word: ${error.message}`;
        } finally {
            setTimeout(() => { deleteLastWordButton.disabled = false; }, 500);
        }   
    });

    // --- NEW GENERATE SENTENCE LOGIC ---
 generateSentenceButton.addEventListener('click', async () => { 
    // 1️⃣ Get recognized words
    const recognizedWords = wordsRecognizedDisplay.textContent.trim().split(" ").filter(w => w !== "");
    
    // 2️⃣ Send to Flask backend
    try {
        const response = await fetch("/generate_sentence", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ words_list: recognizedWords })
        });

        const data = await response.json();

        if (data.error) {
            detectionStatus.textContent = "⚠️ Error generating sentence!";
            console.error(data.error);
            return;
        }

        // 3️⃣ Display the generated sentence
        savedSentenceDisplay.textContent = data.sentence;
        detectionStatus.textContent = "Sentence generated successfully ✅";

        // 4️⃣ Play generated speech (if available)
        if (data.audio_url) {
            const audio = new Audio(data.audio_url);
            audio.play();
        }

    } catch (error) {
        console.error("Error generating sentence:", error);
        detectionStatus.textContent = "⚠️ Error generating sentence!";
    }
});

});