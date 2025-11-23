# Energy Consumption Prediction - Thesis Project

This project analyzes electricity consumption and solar power data, and develops prediction models using machine learning and deep learning techniques.

## 📁 Project Structure

```
bitirme_tezi/
├── notebooks/              # Jupyter notebooks for exploration and analysis
│   ├── grid_client_consumption.ipynb
│   └── nrel_grid_solar_data.ipynb
├── scripts/                # Python scripts for data processing and model training
│   ├── main.py             # Main analysis script
│   ├── client_categorization.py
│   └── lstm_training.py
├── data/                   # Dataset files
│   ├── raw/                # Raw data files (not tracked in git)
│   └── processed/          # Processed/cleaned data
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bitirme_tezi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For Apple Silicon GPU acceleration on macOS:
```bash
pip install tensorflow-metal
```

## 📊 Usage

### Main Analysis Pipeline

Run the complete analysis pipeline:

```bash
python scripts/main.py
```

This script performs:
- Data loading and preprocessing
- Client categorization (Low/High Variance, Sparse Data)
- Cold Start problem detection and trimming
- LSTM model training with Optuna hyperparameter optimization
- Results visualization and summary

### Jupyter Notebooks

For interactive exploration:

```bash
jupyter notebook notebooks/
```

## 🔬 Features

### Data Processing
- **Cold Start Detection**: Automatically identifies and removes leading zeros from client data
- **Client Categorization**: Classifies clients into:
  - Low Variance (Stable): Easy to predict, regular consumers
  - High Variance (Irregular): Challenging consumers where Deep Learning shows potential
  - Sparse Data: Consumers with intermittent data

### Models
- **LSTM**: Long Short-Term Memory networks with Optuna hyperparameter tuning
- Supports Metal GPU acceleration on Apple Silicon Macs

### Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## 📝 Data

The project uses:
- **UCI Electricity Load Diagrams Dataset**: 370 clients, 2011-2014
- **NREL Solar Power Data**: Alabama solar power generation data

**Note**: Large data files are not tracked in git. Place your data files in `data/raw/` directory.

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.8+
- See `requirements.txt` for full list

## 📄 License

[Add your license here]

## 👤 Author

[Your name]

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the electricity consumption dataset
- NREL for solar power data
