# TickerFFT-Analytics

### Multi-Instrument Market Pattern Detection via Rolling FFT and Anomaly Detection

TickerFFT-Analytics is a modular Python pipeline for processing multi-instrument financial time-series data. It computes rolling frequency-domain features using the Fast Fourier Transform (FFT) and detects anomalous market regimes per instrument using unsupervised machine learning.
The project is designed for reproducibility, clear visualisation, and domain-aware analysis of market patterns.

## Features

- **Data Processing**
  - Ingests raw instrument data from structured text files (`data.txt`, `StaticFields.txt`, `DynamicFields.txt`).
  - Separates static and dynamic fields, producing clean CSV outputs for downstream analysis.
  
- **Feature Engineering**
  - Computes rolling FFT features: dominant frequency, total power, and spectral entropy.
  - Handles multi-instrument datasets using sliding windows.
  - Supports per-instrument analysis for better signal isolation.

- **Anomaly Detection**
  - Applies `Isolation Forest` to detect unusual market regimes per instrument.
  - Flags anomalous windows and calculates % anomalous windows for ranking instruments.
  - High-risk instruments are identified automatically.

- **Visualisation**
  - Figures are saved in `charts/` automatically.
  - **Figure 1:** Distribution of dominant frequencies.
  - **Figure 2:** Top 20 instruments by % anomalous windows (high-risk highlighted in red/orange, bolded labels).
  - **Figures 3–7:** Total power over time for the top 5 high-risk instruments, with anomalies highlighted in red.
  - Time axes are consistent across instruments; flattening after ~16:00 reflects market close.

- **Reproducibility**
  - Fully code-driven pipeline; no Jupyter notebooks required.
  - CSV outputs allow inspection and further analysis.
  - `validate_pipeline.py` available for pipeline sanity checks.

## Project Structure

```bash
TickerFFT-Analytics/
│
├── data/                     # Raw input files and processed CSVs
│   ├── data.txt
│   ├── StaticFields.txt
│   ├── DynamicFields.txt
│   └── fft_features*.csv     # Processed features and anomaly outputs
│
├── src/                      # Core Python modules
│   ├── __init__.py
│   ├── InstrumentDataProcessor.py
│   ├── FFTFeatureExtractor.py
│   ├── AnomalyDetector.py
│   ├── dashboard.py          # Financial dashboard generation
│   └── visualization.py      # Plotting functions
│
├── charts/                   # Auto-generated figures (created by main.py)
│   ├── figure_1_dominant_frequency.png
│   ├── figure_2_top_anomalies.png
│   ├── figure_3_total_power_*.png
│   └── ...
│
├── main.py                   # Full pipeline execution
├── validate_pipeline.py      # Pipeline validation / sanity checks
├── requirements.txt          # Python dependencies
└── README.md                 # Project description
```
### Prerequisites
Python 3.11+

Install required packages:
```bash
pip3 install -r requirements.txt
```
Running the Pipeline
```bash
source backend/bin/activate   # or wherever your venv is
python3 main.py
```
This will:
- Process raw instrument data.
- Compute rolling FFT features.
- Run per-instrument anomaly detection.
- Generate financial dashboard and visualisations in charts/.

### Outputs
- fft_features.csv → rolling FFT features per instrument.
- fft_features_with_anomalies.csv → FFT features with anomaly flags.

- charts/ → PNG visualisations:
  - Figure 1: Dominant frequency histogram.
  - Figure 2: Top 20 anomalous instruments bar chart.
  - Figures 3–7: Total power over time for top 5 high-risk instruments.

### Time Axis Notes
The X-axis in total power plots may flatten or drop after ~16:00.

This reflects market close, not a bug or anomaly in the pipeline.

### Using Synthetic Data
For testing or demonstration without live market data:

```python
from src.utils import generate_synthetic_ticker_data

df = generate_synthetic_ticker_data(
    instruments=["AAPL", "GOOG", "MSFT"],
    start_date="2025-01-01",
    num_days=100
)
```
### Technical Highlights
- Data Engineering: Parsing structured text, separating dynamic and static fields, producing machine-learning-ready CSVs.
- Feature Engineering: Rolling FFT, total power, spectral entropy, per-instrument windows.
- Machine Learning: Isolation Forest anomaly detection, high-risk scoring.
- Visualisation: Clear, reproducible plots highlighting anomalies and high-risk instruments.

### Potential Extensions
- Integration with quantitative backtesting frameworks.
- Interactive dashboards using Streamlit or Dash.
- Multi-instrument heatmaps of dominant frequencies or spectral entropy.
- Academic comparisons (e.g., classical vs quantum FFT).

### License
MIT License