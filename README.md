# House Price Prediction ML Pipeline

Intelligent Property Valuation Systems (IPVS) represent cutting-edge technological solutions designed to calculate real estate property market values through advanced computational techniques. These intelligent platforms harness sophisticated machine learning algorithms and expansive property information repositories to generate automated property value assessments. The fundamental objective of an IPVS is to deliver rapid, impartial, and economically efficient property worth evaluations across various temporal contexts.

The primary operational mechanism of an IPVS involves analyzing transactional data from recently sold properties within specific geographical markets. By systematically evaluating multidimensional attributes such as geographical positioning, structural characteristics, property condition, and contextual economic indicators, the system can identify statistically relevant property comparables. Through complex algorithmic processes, these systems extrapolate probable market values by examining intricate relationships between comparable property attributes and their historical transaction prices. Advanced IPVS platforms may integrate supplementary data streams, including professional valuation archives and macroeconomic trend analyses, to refine estimation precision.

Algorithmic property valuation systems present numerous advantages over conventional property assessment methodologies. These technological solutions enable dramatically accelerated valuation processes with significantly reduced operational costs. Moreover, they strive to eliminate subjective human interpretations by implementing standardized computational evaluation frameworks. However, these systems are not universally applicable, particularly when confronting properties exhibiting extraordinary or atypical characteristics that deviate from standard market patterns.
Fundamentally, machine learning-powered property valuation platforms have emerged as transformative instruments within the real estate ecosystem, facilitating more expeditious, cost-effective, and data-driven property assessment strategies across diverse transactional scenarios. The objective remains constructing a robust IPVS model adhering to predefined specifications, utilizing provided datasets and maintaining prescribed project architectural configurations.

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
