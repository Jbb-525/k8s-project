"""
MNIST Inference Service - PyTorch Version
Provides REST API for handwritten digit recognition
"""
from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import numpy as np
import base64
import io
from PIL import Image
import os
import sys

app = Flask(__name__)

# Global variables for model
model = None
device = None

# Define model architecture (must match training!)
class MNISTNet(nn.Module):
    """Simple CNN model for MNIST classification"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# HTML Template - User Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Handwritten Digit Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .tech-badge {
            text-align: center;
            margin: 10px 0;
            color: #666;
            font-size: 14px;
        }
        .tech-badge span {
            background-color: #e3f2fd;
            padding: 5px 15px;
            border-radius: 15px;
            margin: 0 5px;
        }
        .upload-section {
            margin: 30px 0;
            padding: 20px;
            border: 2px dashed #ccc;
            border-radius: 5px;
            text-align: center;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #e8f5e9;
            border-radius: 5px;
            display: none;
        }
        .result h2 {
            color: #2e7d32;
            margin-top: 0;
        }
        .prediction {
            font-size: 48px;
            font-weight: bold;
            color: #1b5e20;
            text-align: center;
            margin: 20px 0;
        }
        .confidence {
            text-align: center;
            color: #666;
        }
        .preview {
            text-align: center;
            margin: 20px 0;
        }
        .preview img {
            max-width: 200px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .all-predictions {
            margin-top: 20px;
            font-size: 14px;
        }
        .prediction-bar {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        .bar {
            height: 20px;
            background-color: #4CAF50;
            border-radius: 3px;
            margin-left: 10px;
        }
        .info {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .info h3 {
            margin-top: 0;
            color: #1976d2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔢 MNIST Handwritten Digit Recognition</h1>
        <div class="tech-badge">
            <span>🔥 PyTorch</span>
            <span>🐳 Docker</span>
            <span>☸️ Kubernetes</span>
        </div>
        
        <div class="info">
            <h3>📌 Instructions:</h3>
            <ul>
                <li>Upload a handwritten digit image (0-9)</li>
                <li>Supports PNG, JPG, JPEG formats</li>
                <li>The system will automatically recognize the digit</li>
                <li>Powered by PyTorch CNN model</li>
            </ul>
        </div>
        
        <div class="upload-section">
            <h3>Upload Image</h3>
            <input type="file" id="imageInput" accept="image/*" onchange="previewImage()">
            <br><br>
            <button onclick="predictImage()">🔍 Recognize Digit</button>
        </div>
        
        <div class="preview" id="preview" style="display:none;">
            <h3>Preview:</h3>
            <img id="previewImg" src="" alt="Preview">
        </div>
        
        <div class="result" id="result">
            <h2>Prediction Result:</h2>
            <div class="prediction" id="prediction">-</div>
            <div class="confidence" id="confidence">-</div>
            <div class="all-predictions" id="allPredictions"></div>
        </div>
    </div>
    
    <script>
        function previewImage() {
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const previewImg = document.getElementById('previewImg');
            
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        function predictImage() {
            const input = document.getElementById('imageInput');
            if (!input.files || !input.files[0]) {
                alert('Please select an image first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', input.files[0]);
            
            // Show loading state
            document.getElementById('prediction').textContent = 'Recognizing...';
            document.getElementById('result').style.display = 'block';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Display main prediction result
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('confidence').textContent = 
                    `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                
                // Display all class probabilities
                let allPredHtml = '<h4>All Digit Probabilities:</h4>';
                for (let i = 0; i < data.all_predictions.length; i++) {
                    const prob = data.all_predictions[i];
                    const barWidth = prob * 100;
                    allPredHtml += `
                        <div class="prediction-bar">
                            <span style="width:30px; display:inline-block;">${i}:</span>
                            <div class="bar" style="width:${barWidth}%"></div>
                            <span style="margin-left:10px;">${(prob * 100).toFixed(2)}%</span>
                        </div>
                    `;
                }
                document.getElementById('allPredictions').innerHTML = allPredHtml;
                document.getElementById('result').style.display = 'block';
            })
            .catch(error => {
                alert('Prediction failed: ' + error);
            });
        }
    </script>
</body>
</html>
"""

def load_model():
    """Load trained model"""
    global model, device
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = '/mnt/model-storage/mnist_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run training job first!")
        return False
    
    print(f"Loading model from: {model_path}")
    
    # Load complete model (structure + weights)
    model = torch.load(model_path, map_location=device)
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    
    # Read model info
    info_path = '/mnt/model-storage/model_info.txt'
    if os.path.exists(info_path):
        print("\nModel Information:")
        with open(info_path, 'r') as f:
            print(f.read())
    
    return True

def preprocess_image(image):
    """Preprocess image to match model input"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Invert colors (MNIST is white digits on black background)
    img_array = 1.0 - img_array
    
    # Standardize (using MNIST mean and std)
    mean = 0.1307
    std = 0.3081
    img_array = (img_array - mean) / std
    
    # Convert to PyTorch tensor and add batch and channel dimensions (1, 1, 28, 28)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

@app.route('/')
def home():
    """Home page - display upload interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'unhealthy', 'message': 'Model not loaded'}), 503
    return jsonify({'status': 'healthy', 'message': 'Service is running', 'framework': 'PyTorch'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint - receive image and return prediction result"""
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        processed_image = processed_image.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(processed_image)
            # Use softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
        
        # Get prediction result
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        all_predictions = probabilities.tolist()
        
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }
        
        print(f"Prediction result: digit {predicted_class}, confidence: {confidence:.4f}")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_number', methods=['POST'])
def predict_number():
    """
    Alternative API - receive JSON format image data
    Accepts format: {"image": "base64 encoded image"}
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'Missing image field'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        processed_image = processed_image.to(device)
        
        with torch.no_grad():
            output = model(processed_image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Starting MNIST Inference Service (PyTorch)")
    print("=" * 50)
    
    # Load model
    if not load_model():
        print("Warning: Model not loaded, service will start but cannot make predictions")
    
    # Start Flask service
    print("\nService running on port 5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)