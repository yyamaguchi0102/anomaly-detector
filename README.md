# Real-Time Anomaly Detection System

A lightweight, real-time anomaly detection tool for cybersecurity logs using machine learning.

**Full Technical Documentation** → [View on Notion](https://southern-feta-f21.notion.site/Real-Time-Anomaly-Detection-Doc-20e738750e7a80feb78cc750018ef78f)**
**日本語技術ドキュメントはこちら** → [Notionページを開く](https://southern-feta-f21.notion.site/Doc-20e738750e7a8010b536c967ad0aceb4)**

## Project Structure
```
.
├── data/
│   ├── raw/          # Original, immutable data
│   └── processed/    # Cleaned and processed data
├── src/
│   ├── data/         # Data loading and processing scripts
│   ├── features/     # Feature engineering scripts
│   ├── models/       # Model training and prediction scripts
│   └── utils/        # Utility functions
└── tests/            # Test files
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Generate sample data:
```bash
python src/data/generate_sample_data.py
```

2. Train the model:
```bash
python src/models/train.py
```

3. Run anomaly detection:
```bash
python src/models/detect.py
```

## Features
- Real-time log analysis
- Isolation Forest-based anomaly detection
- Feature engineering for temporal patterns
- Modular and extensible architecture

## Future Enhancements
- Autoencoder/LSTM models
- Streamlit dashboard
- Real-time alerting system
- Integration with syslog
