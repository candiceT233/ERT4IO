# Results for 1GB Data Size per Task Tests

This folder contains test results for I/O performance evaluations using a 1GB data size per task. The tests were conducted on different node counts and include both regular IOR results and checkpoint (cp) results.

## Folder Contents Overview

- **Gateway result folders:**
  - `1n_gateway_results_1000bs/`
  - `2n_gateway_results_1000bs/`
  - `4n_gateway_results_1000bs/`
  - `8n_gateway_results_1000bs/`
  - `1n_gateway_cp_results_1000bs/`
  - `2n_gateway_cp_results_1000bs/`
  - `4n_gateway_cp_results_1000bs/`
  - `8n_gateway_cp_results_1000bs/`

- **CSV files for best bandwidth measurements:**
  - Files prefixed with `best_bandwidth_1gb_pertask_` followed by node count and operation type (e.g., `1n_best_read.csv`, `8n_best_write.csv`, `4n_cp.csv` etc.)

- **Other files:**
  - `bandwidth_performance_data.csv` — aggregated performance data summary
  - `aggregate_darshan_counters.py` — Python script to aggregate Darshan I/O statistics
  - Jupyter notebooks for visualization:
    - `plot_cp_performance.ipynb` — plots for checkpoint (cp) results only
    - `plot_performance.ipynb` — plots for regular IOR results

## Usage Instructions

### Aggregate Darshan Statistics

To aggregate Darshan counters from the raw gateway results, use the included Python script `aggregate_darshan_counters.py`. It supports the following folders containing raw Darshan logs:

- `1n_gateway_results_1000bs`
- `2n_gateway_results_1000bs`
- `4n_gateway_results_1000bs`
- `8n_gateway_results_1000bs`

Run the script from this directory as follows:

```bash
python3 aggregate_darshan_counters.py 1n_gateway_results_1000bs
python3 aggregate_darshan_counters.py 2n_gateway_results_1000bs
python3 aggregate_darshan_counters.py 4n_gateway_results_1000bs
python3 aggregate_darshan_counters.py 8n_gateway_results_1000bs
