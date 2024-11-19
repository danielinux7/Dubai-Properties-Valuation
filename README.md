# House Price Prediction Model

A machine learning model for predicting house prices using ensemble methods including XGBoost, Random Forest, and Support Vector Regression.

## Table of Contents
- [Data Requirements](#data-requirements)
- [Installation](#installation)
  - [Python Setup](#python-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
  - [Inference Mode](#inference-mode)
  - [Training Mode](#training-mode)
  - [Test Mode](#test-mode)

## Data Requirements

Each datapoint should contain the following fields. Use "nan" for unavailable values. See `sample.json` in the data folder for examples.

Required fields:
- `amount`
- `total_buyer`
- `total_seller`
- `transaction_size_sqm`
- `property_size_sqm`
- `rooms_en`
- `transaction_type_en`
- `transaction_subtype_en`
- `registration_type_en`
- `is_freehold_text`
- `property_usage_en`
- `property_type_en`
- `property_subtype_en`
- `project_name_en`
- `area_en`
- `nearest_landmark_en`
- `nearest_metro_en`
- `nearest_mall_en`

## Installation

1. Navigate to the root directory
2. Unzip data(s) and model(s):
   ```bash
   cd data/ && unzip snp_dld_2024_transactions.zip
   cd data/ && unzip snp_dld_2024_rents.zip
   cd models/ && unzip rf_model.zip
   ```
### Python Setup

1. Navigate to the root directory
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup

Ensure Docker is installed, then build the image:
```bash
docker build -t <IMAGE_NAME> .
```

## Usage

The program supports three modes: inference, training, and testing. All JSON files should be placed in the `data` folder.

Common arguments:
- `--mode`: Choose between "train", "inference", or "test" (default: "inference")
- `--json_file`: Input data file (default: "sample.json")
- `--config`: Configuration file (default: "config.json")

### Inference Mode

Use this mode to make predictions using a trained model.

**Python:**
```bash
# Using defaults
python main.py

# With custom options
python main.py --mode inference --json_file sample.json
```

**Docker:**
```bash
# Using defaults
docker run <IMAGE_NAME>

# With custom options (Not supported to pass json file yet, behaves as defualts)
docker run <IMAGE_NAME> --mode inference --json_file sample.json
```

### Training Mode

Use this mode to train the model. The config file allows fine-tuning of Bayesian optimization parameters for XGBoost, Random Forest, Support Vector Regressor, and Meta Learner.

**Python:**
```bash
# Using defaults
python main.py --mode train

# With custom config
python main.py --mode train --config config.json
```

**Docker:**
```bash
# Using defaults
docker run <IMAGE_NAME> --mode train

# With custom config (Not supported to pass config file yet, behaves as defualts)
docker run <IMAGE_NAME> --mode train --config config.json
```

### Test Mode

Runs unit tests through the complete data pipeline including training and inference.

**Python:**
```bash
# Using defaults
python main.py --mode test

# With custom config
python main.py --mode test --config config.json
```

**Docker:**
```bash
# Using defaults
docker run <IMAGE_NAME> --mode test

# With custom config (Not supported to pass config file yet, behaves as defaults)
docker run <IMAGE_NAME> --mode test --config config.json
```
