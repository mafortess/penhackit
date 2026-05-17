## Evaluated models and metrics

| Model | Type | Main metrics | Reason for inclusion |
|---|---|---|---|
| Decision Tree | Interpretable supervised model | Accuracy, F1-score, confusion matrix | Simple and explainable baseline. |
| Random Forest | Ensemble tabular model | Accuracy, F1-score, valid action rate | More robust than a single decision tree. |
| MLP | Dense neural network | Accuracy, F1-score, per-class errors | Tests a non-linear neural approach. |
| CatBoost / XGBoost / LightGBM | Gradient boosting model | Accuracy, F1-score, confusion matrix | Strong baseline for tabular data. |