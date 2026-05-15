# HDVF Python Validation

This repository contains the Python validation scripts for the HDVF paper:
- `hdvf_synthetic_recovery.py`: Synthetic HDVF validation (Heston recovery and leverage regimes)
- `hdvf_indian_empirical.py`: Indian 1-minute empirical HDVF extension

## Setup

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the synthetic recovery script via:
```bash
python3 hdvf_synthetic_recovery.py
```

You can run the Indian empirical script via:
```bash
python3 hdvf_indian_empirical.py
```
