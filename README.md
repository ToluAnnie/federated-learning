# Federated Learning for Privacy-Preserving Diabetes Risk Prediction

## Overview

This project implements a federated learning system using [Flower](https://flower.dev/) and [PyTorch](https://pytorch.org/) to train a predictive healthcare model for diabetes risk prediction. The system follows a decentralized approach where:

* A global model is initialized on a central server
* Clients train the model using their private data (simulated synthetic health data)
* Model updates (gradients/weights) are aggregated with differential privacy
* Patient data never leaves the device/hospital

## Privacy Enhancements

* **Differential Privacy**: Implemented in `privacy/differential_privacy.py` to protect model updates
* **Data Isolation**: All client data remains local and never transmitted
* **Secure Aggregation**: Server only receives model updates, not raw patient data

## Use Case

This system simulates a real-world scenario where hospitals or health apps train a predictive model on sensitive patient data without sharing that data. The model predicts a patient’s likelihood of developing diabetes based on features such as age, BMI, glucose level, and blood pressure.

## Example Prediction

Example patient data:

```python
{"age": 50, "bmi": 30.1, "blood sugar": 150, "family history": 'yes', "physical activity": 'no', }
````

Predicted Output:

```python
Risk Score: 0.76 → Likely diabetic
```

## Data Description

Each client uses a synthetic healthcare dataset with the following features:

  * Age
  * BMI (Body Mass Index)
  * Blood Sugar
  * Family History
  * Physical Activity

## Architecture

  * The server coordinates training and aggregates model updates.
  * Each client trains on its own local dataset.
  * No raw data is shared; only encrypted model updates are exchanged.

## Prerequisites

### Software

  * [Python 3.11+](https://www.python.org/downloads/)
  * [Git](https://git-scm.com/)

### Python Dependencies

Install in your project directory:

```bash
pip install torch flwr numpy flower-superlink diffprivlib
```

## File Structure

```
├── data/
│   └── synthetic_health_data.py      # Synthetic healthcare dataset generator
├── models/
│   └── federated_logistic_regression.py  # PyTorch model definition
├── privacy/
│   └── differential_privacy.py       # Differential privacy implementation
├── client.py                         # Federated learning client implementation
├── server.py                         # Centralized federated learning server
├── README.md                         # This file
├── .gitignore                        # Git version control configuration
└── requirements.txt                  # Python dependencies
```

## Setup and Execution

### 1\. Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: Federated learning diabetes prediction system"
```

### 2\. Create GitHub Repository

1.  Go to [GitHub New Repository](https://github.com/new)
2.  Create a new repository (e.g., `federated-diabetes-prediction`)

### 3\. Push to GitHub

```bash
git remote add origin [https://github.com/your-username/repo-name.git](https://github.com/your-username/repo-name.git)
git push -u origin main
```

### 4\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5\. Run the System

1.  **Start the central server**:

<!-- end list -->

```bash
python server.py
```

2.  **Run clients in separate terminals**:

<!-- end list -->

```bash
python client.py --client-id 1
python client.py --client-id 2
python client.py --client-id 3
```

  - Each client trains the model on synthetic data and sends updates to the server.
  - Verify server logs for confirmation of client connection events.

## Testing

```bash
python test_federated_learning.py
```

## Model Details

  * Logistic regression model for binary classification (diabetes risk)
  * BCELoss for binary cross-entropy
  * SGD optimizer with learning rate 0.01
  * Federated Averaging (FedAvg) strategy

## Limitations and Future Work

  * Synthetic data may not fully represent real patient variation

  * Secure aggregation with cryptographic primitives not yet integrated

  * Model may underperform in highly imbalanced datasets

  * Future plans:

      * Integrate secure aggregation protocols
      * Deploy real-world datasets (de-identified and consented)
      * Optimize model for edge deployment

## Contributing

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/your-feature`)
3.  Commit your changes (`git commit -am 'Add some feature'`)
4.  Push to the branch (`git push origin feature/your-feature`)
5.  Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Troubleshooting

### Missing Dependencies

```bash
pip install torch flwr numpy flower-superlink diffprivlib
```

### Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

```
```
