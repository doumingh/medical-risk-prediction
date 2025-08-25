# Medical Risk Prediction System


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-risk-prediction.git
   cd medical-risk-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Access the system**
   Open your browser and visit: [http://localhost:8001](http://localhost:8001)


## 📖 Usage

### 1. Input Patient Data
- **Age**: Patient's age (1-120 years)
- **Number of Polyps**: Count of polyps (0-100)
- **Long Diameter**: Maximum dimension in mm (0.1-100)
- **Short Diameter**: Minimum dimension in mm (0.1-100)
- **Base Type**: Pedicle or Broad base

### 2. Submit for Prediction
Click the "Start Prediction" button to analyze the data

### 3. View Results
- **Risk Level**: Color-coded risk classification
- **Probability**: Risk percentage with confidence
- **Treatment Recommendations**: Evidence-based medical advice

### 4. Analyze Features
Explore SHAP charts for feature importance analysis

## 🎯 Risk Levels

| Risk Level | Color | Description |
|------------|-------|-------------|
| **Low Risk** | 🟢 Green | Follow-up not required |
| **Moderate Risk** | 🟡 Yellow | Regular monitoring recommended |
| **High Risk** | 🟠 Orange | Surgical intervention considered |
| **Very High Risk** | 🔴 Red | Immediate surgery strongly recommended |

