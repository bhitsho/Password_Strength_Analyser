# Password Strength Analyzer

## Overview
This project is a **Password Strength Analyzer** built using a Recurrent Neural Network (RNN). It evaluates password strength by training on a dataset and predicting the robustness of new inputs. The RNN model also generates strong passwords with high entropy to improve security.

## Features
- **Password Strength Prediction:** Uses a trained RNN model to classify password strength.
- **Interactive Web Interface:** Powered by Flask for user-friendly interaction.
- **Visual Explanation Tool:** Through `xaibarclays.ipynb`, providing insight into model predictions.
- **Model Training Pipeline:** Simplified pipeline for training and evaluating new models.
- **Scalable Architecture:** Easily extendable to include more complex architectures like Transformers.

## Files
- `app.py`: Flask application for running the model and serving the web interface.
- `password_rnn_pipeline.h5`: Trained RNN model. [Download Model](https://drive.google.com/file/d/1Ryrpf1Fuy-cihvvRQhQtFBPKhzcV5yK6/view?usp=drive_link)
- `requirements.txt`: Dependencies required to run the project.
- `something.py`: Additional scripts for processing or testing.
- `xaibarclays.ipynb`: Jupyter notebook for visual explanations and model analysis.
- `templates/`: HTML templates used by the Flask application.

## Installation
### Prerequisites
- Python 3.8 or above
- Flask
- TensorFlow / Keras
- Jupyter Notebook (for analysis and visualization)

### Installation Steps
1. **Clone the repository:**
```bash
   git clone <repository-link>
```

2. **Navigate to the project directory:**
```bash
   cd Password-Strength-Analyzer
```

3. **Install the dependencies:**
```bash
   pip install -r requirements.txt
```

## Usage
1. **Run the Flask application:**
```bash
   python app.py
```

2. **Open your browser and go to:**
```
   http://127.0.0.1:5000/
```

## Model Training
The model was trained on a dataset of passwords labeled with varying strengths. The RNN architecture includes LSTM layers to capture sequential dependencies and improve prediction accuracy.

### Training Procedure
- Preprocessing: Tokenization, padding, and embedding.
- Model Architecture: LSTM-based network with dropout layers to prevent overfitting.
- Evaluation: Accuracy, precision, recall, and F1-score metrics.
- Saving the Model: Saved as `password_rnn_pipeline.h5`.

## Future Improvements
- Enhance model performance by training on a larger, more diverse dataset.
- Integrate advanced architectures such as Transformers and attention mechanisms.
- Deploy as a web service using Docker or cloud platforms.
- Implement a password suggestion feature for generating secure passwords.
- Improve visualization tools for better explainability.

## Authors
- **CaptHeisenberg**
- **ayushpathak477**
- **bhitsho**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by various password strength estimation methodologies and neural network architectures.

