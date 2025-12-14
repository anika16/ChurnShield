# ChurnShield - AI-Based Customer Churn Prediction

Quick start:
1. Create virtual env and install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Place IBM Telco Customer Churn CSV at:
   data/raw/Telco-Customer-Churn.csv

3. Run preprocessing:
   python src/scripts/run_preprocess.py

4. Train model:
   python src/scripts/train_pipeline.py

5. Start API:
  RUN E:\ChurnShield\ChurnShield>  python -m api.app

6. Start dashboard:
   Inside ChurnShield\ChurnShield> python -m streamlit run dashboard/dashboard_app.py
