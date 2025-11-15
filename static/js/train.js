// static/js/train.js
document.addEventListener('DOMContentLoaded', function() {
    const trainModelButton = document.getElementById('trainModelButton');
    const trainingStatus = document.getElementById('trainingStatus');

    trainModelButton.addEventListener('click', async () => {
        trainingStatus.textContent = "Training model... This may take a while, please wait.";
        trainModelButton.disabled = true;

        try {
            const response = await fetch('/start_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();

            if (data.status === 'success') {
                trainingStatus.textContent = `Training complete! Model accuracy: ${data.accuracy.toFixed(2)}. You can now go to the "Detect" page.`;
            } else {
                trainingStatus.textContent = `Error: ${data.message}`;
            }
        } catch (error) {
            trainingStatus.textContent = `An error occurred: ${error.message}. Check server console.`;
            console.error("Training request failed:", error);
        } finally {
            trainModelButton.disabled = false;
        }
    });
});