# TickerFFT-Analytics

### Market Pattern Detection via Fourier Transform and Anomaly Detection

This project demonstrates a modular Python pipeline for multi-instrument financial time-series data. It extracts rolling frequency-domain features using the Fast Fourier Transform (FFT) and detects anomalous market regimes using unsupervised machine learning.


## Features

- **Data Processing**
  - Ingests raw instrument data from structured text files (`data.txt`, `StaticFields.txt`, `DynamicFields.txt`).
  - Separates static and dynamic fields, and outputs a clean CSV for downstream analysis.
  
- **Feature Engineering**
  - Computes rolling FFT features, including dominant frequency, total power, and spectral entropy.
  - Handles multi-instrument datasets with sliding windows.
  - Supports optional computation of log-returns and volatility features.

- **Anomaly Detection**
  - Uses `Isolation Forest` to detect unusual market regimes per instrument.
  - Supports anomaly scoring for enhanced visualisation and analysis.

- **Visualization**
  - Plots FFT-derived total power over time, highlighting normal vs anomalous regimes.
  - Log-scaling for total power ensures extreme volatility events remain readable.
  - Can be extended to interactive dashboards using Streamlit or Dash.

- **Synthetic Data Support**
  - Provides a utility to generate reproducible multi-instrument synthetic time-series data for testing and demonstration.

## Project Structure

```bash
TickerFFT-Analytics/
│
├── data/ # Input/output files
│ ├── data.txt
│ ├── StaticFields.txt
│ ├── DynamicFields.txt
│ └── output_*.csv # Processed CSV outputs
│
├── src/ # Main Python modules
│ ├── init.py
│ ├── InstrumentDataProcessor.py
│ ├── FFTFeatureExtractor.py
│ ├── AnomalyDetector.py
│ └── utils.py # Synthetic data generator and helpers
│
├── notebooks/ # Jupyter notebooks
│ └── fft_visualisation.ipynb
│
├── tests/ # Unit tests
│ ├── test_data_processor.py
│ ├── test_fft_extractor.py
│ └── test_anomaly_detector.py
│
├── requirements.txt # Python dependencies
├── README.md # Project description
└── main.py # Example pipeline script
```

## Getting Started

### Prerequisites

- Python 3.11+ recommended
- Install required packages:

```bash
pip install -r requirements.txt
Running the Pipeline
Process raw ticker data:
```
```bash
python main.py
```
Compute FFT features and detect anomalies (handled automatically in main.py).

### View results:

- FFT features are saved as fft_features.csv.
- Anomaly detection results are saved as fft_features_with_anomalies.csv.
- Visualisations can be generated in notebooks/fft_visualisation.ipynb.

### Using Synthetic Data
For testing or demonstration without real market data:
```bash
from src.utils import generate_synthetic_ticker_data

df = generate_synthetic_ticker_data(
    instruments=["AAPL", "GOOG", "MSFT"],
    start_date="2025-01-01",
    num_days=100
)
```
### Technical Highlights

**Data Engineering**: Parsing raw structured text files, handling dynamic and static fields, pivoting for ML pipelines.

**Feature Engineering**: FFT-based rolling window feature extraction; log-return and volatility support.

**Machine Learning**: Unsupervised anomaly detection via Isolation Forest; per-instrument anomaly scoring.

**Visualisation**: Clear plotting of market regimes using log-scaled FFT power; supports multi-instrument dashboards.

### Potential Extensions
- Integration with quantitative backtesting frameworks.
- Classical → Quantum FFT comparison for academic or research purposes.
- Interactive dashboards using Streamlit or Dash.
- Multi-instrument heatmaps of dominant frequencies or spectral entropy.

### License
MIT License.

