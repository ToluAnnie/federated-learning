# Federated Learning for Privacy-Preserving Diabetes Risk Prediction

## Overview
This project implements a federated learning system using [Flower](https://flower.dev/) and [PyTorch](https://pytorch.org/) to train a predictive healthcare model for diabetes risk prediction. The system follows a decentralized approach where:
- A global model is initialized on a central server
- Clients train the model using their private data (simulated synthetic health data)
- Model updates (gradients/weights) are aggregated with differential privacy
- Patient data never leaves the device/hospital

## Prerequisites
### Software
- [Python 3.11+](https://www.python.org/downloads/) (Install via [official installer](https://www.python.org/downloads/))
- [Git](https://git-scm.com/) (Install via [official installer](https://git-scm.com/))

### Python Dependencies
Install in your project directory:
```bash
pip install torch flwr numpy
```

## File Structure
```
├── data/
│   └── synthetic_health_data.py      # Synthetic healthcare dataset generator
├── models/
│   └── federated_logistic_regression.py  # PyTorch model definition
├── privacy/
│   └── differential_privacy.py       # Differential privacy implementation
├── clients/
│   └── client.py                     # Federated learning client implementation
├── server/
│   └── server.py                     # Centralized federated learning server
├── centralized.py                    # Central server logic with FedAvg strategy
├── README.md                         # This file
├── .gitignore                        # Git version control configuration
└── requirements.txt                  # Python dependencies
```

## Setup and Execution

### 1. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Federated learning diabetes prediction system"
```

### 2. Create GitHub Repository
1. Go to [GitHub New Repository](https://github.com/new)
2. Create a new repository (e.g., `federated-diabetes-prediction`)

### 3. Push to GitHub
```bash
git remote add origin https://github.com/your-username/repo-name.git
git push -u origin main
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the System
1. **Start the central server**:
```bash
python server.py
```

2. **Run clients in separate terminals**:
```bash
python client.py 1  # Client 1
python client.py 2  # Client 2
python client.py 3  # Client 3
```

## Privacy Enhancements
- **Differential Privacy**: Implemented in `privacy/differential_privacy.py` to protect model updates
- **Data Isolation**: All client data remains local and never transmitted
- **Secure Aggregation**: Server only receives model updates, not raw patient data

## Testing
```bash
python tests/test_federated_learning.py
```

## Model Details
- Logistic regression model for binary classification (diabetes risk)
- BCELoss for binary cross-entropy
- SGD optimizer with learning rate 0.01
- Federated Averaging (FedAvg) strategy

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting
### Missing Dependencies
If you see errors like `ModuleNotFoundError`, ensure you've installed all dependencies:
```bash
pip install torch flwr numpy
```

### Virtual Environment
It's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Differential Privacy
If you need additional privacy libraries:
```bash
pip install opacus diffprivlib