    
async function detectFakeNews() {
    const text = document.getElementById('newsText').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
    });
    const result = await response.json();
    const resultElement = document.getElementById('result');
    if (result.error) {
        resultElement.innerText = 'Error: ' + result.error;
        resultElement.style.color = '#FF512F';
    } else {
        resultElement.innerText = 'Prediction: ' + (result.prediction === 1 ? 'Fake' : 'Real');
        resultElement.style.color = result.prediction === 1 ? '#FF512F' : '#72EDF2';
    }
    if (!text) {
        document.getElementById('result').innerText = 'Please enter some text to analyze.';
        document.getElementById('result').style.color = '#FF512F';
        return;
    }
    
}
 