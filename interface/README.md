# KneeFwd

KneeFwd is an academic-style simulator interface for exploring post-knee-replacement recovery trajectories and missing wearable-data patterns.

This version provides the interface only. Simulation algorithms, missingness mechanisms, imputation models, and accuracy calculations are intentionally deferred.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app opens a Streamlit interface where you can configure a simulated patient cohort, select a clinical recovery trajectory, choose missing-data mechanisms, and inspect placeholder output panels for future simulator results.
