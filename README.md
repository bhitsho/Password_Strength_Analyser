# Password Strength Analyzer 

## Overview
This project is a prototype Password Strength Analyzer built using a Recurrent Neural Network (RNN). It is designed to evaluate the strength of passwords by training on a dataset and predicting the strength of new inputs. It also generates a new strong password with high entropy using a RNN architecture. 

## Features
- **Password Strength Prediction** using a trained RNN model.
- **Interactive Web Interface** powered by Flask.
- **Visual Explanation Tool** (via `xaibarclays.ipynb`) for understanding model predictions.

## Files
- [Download Model (password_rnn_pipeline.h5)](https://drive.google.com/file/d/1Ryrpf1Fuy-cihvvRQhQtFBPKhzcV5yK6/view?usp=sharing)
- `app.py`: Flask application for running the model and serving the web interface.
- `password_rnn_pipeline.h5`: Trained RNN model.
- `requirements.txt`: Dependencies required to run the project.
- `something.py`: Additional scripts for processing or testing.
- `xaibarclays.ipynb`: Jupyter notebook for visual explanations and model analysis.
- `templates/`: HTML templates used by the Flask application.

## Installation
1. Clone the repository:
```bash
   git clone https://github.com/bhitsho/Password_Strength_Analyser
```

2. Navigate to the project directory:
```bash
   cd Password-Strength-Analyzer
```

3. Install the dependencies:
```bash
   pip install -r requirements.txt
```

## Usage
1. Run the Flask application:
```bash
   python app.py
```

2. Open your browser and go to:
```
   http://127.0.0.1:5000/
```

## Model Training
The model was trained on a dataset of passwords with various strengths labeled. The RNN architecture used includes LSTM layers to capture sequential dependencies.

## Future Improvements
- Enhance model performance by training on a larger, more diverse dataset.
- Integrate advanced architectures like Transformers.
- Deploy as a web service using Docker.


## Authors
- **CaptHeisenberg**
- **bhitsho**
- **ayushpathak477**

---

Team Heisenberg

