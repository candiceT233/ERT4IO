# S3 Latency Analysis Tools

This directory contains Python scripts for analyzing I/O latency patterns from AWS S3 benchmark results.

## Scripts Overview

### Extract and Process Results
Scripts for extracting and processing raw benchmark data:
- `process_raw_results.py` - Processes raw benchmark output files
- `extract_latency_metrics.py` - Extracts latency metrics from processed data

### Visualization Tools
Jupyter notebooks for data visualization:
- `latency_plots.ipynb` - Generates latency distribution plots
- `time_series_analysis.ipynb` - Time series analysis of latency patterns

### Analysis Utilities
Supporting analysis scripts:
- `statistics_utils.py` - Statistical analysis functions
- `data_cleaning.py` - Data cleaning and preprocessing utilities

## Usage

1. Process raw benchmark results:
```bash
python3 process_raw_results.py <input_directory> <output_file>
```

2. Extract latency metrics:
```bash
python3 extract_latency_metrics.py <processed_data> <output_file>
```

3. Run Jupyter notebooks for visualization:
```bash
jupyter notebook latency_plots.ipynb
```

## Data Format

Input data should be in CSV format with the following columns:
- Timestamp
- Operation Type (READ/WRITE)
- Transfer Size
- Latency (ms)
- Status Code

## Dependencies

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

Install dependencies:
```bash
pip install -r requirements.txt
```

## Output

The analysis generates:
- Latency distribution plots
- Statistical summaries
- Time series visualizations
- Aggregated metrics by operation type