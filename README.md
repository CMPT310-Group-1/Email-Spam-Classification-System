# Email Spam Classification System
CMPT 310 Group 1 | Email Classifier

## Team Members
- Ngoc Gia Bao Nguyen (301657240) - bnn@sfu.ca
- Jisoo Im (301376613) - jia24@sfu.ca
- James Park (301398024) - hpa55@sfu.ca
- Kevin Han (301610266) - kha166@sfu.ca

## Milestone 1: Binary Classifier (July 1)
- Datasets used:  
`https://github.com/Apaulgithub/oibsip_taskno4/blob/main/spam.csv`  
`https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset`
- Combined dataset with `combine.py`.
- Preprocessed raw dataset.
- Deployed Stochastic Gradient Descent using a Logistic Loss function and L2 Regularization.
- Exported the model as `stage1.pkl` for future milestone.

## Setup
**1. Create a virtual environment:**
- `python -m venv venv`

**2. Activate the environment:**
- **Windows (Git Bash):** `source venv/Scripts/activate`
- **Mac/Linux:** `source venv/bin/activate`

**3. Install the dependencies:**
- `pip install -r requirements.txt`

**4. Run Stage 1 classifier:**
- `python model1.py`
