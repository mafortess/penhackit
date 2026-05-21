# Models

This directory stores the trained decision models used by PenHackIt to select the next semantic action during autonomous or suggestion sessions.

The models are trained from Behavioral Cloning datasets generated from observed pentesting sessions. Each dataset sample represents a transition of the form:

```text
state_t -> action_id
```

Where:

state_t is a fixed-size representation derived from the session Knowledge Base.
action_id is the semantic action selected or executed at that step.
The model learns to predict the next action from the current state.

The models stored here are not responsible for building commands directly. They only predict semantic action identifiers. Command generation, placeholder resolution, execution, parsing, event generation and KB updates are handled by deterministic components of the system.

Directory structure

Each trained model is stored in its own subdirectory:

models/
├── decision_tree/
│   ├── model.joblib
│   └── metrics.json
├── random_forest/
│   ├── model.joblib
│   └── metrics.json
├── catboost/
│   ├── model.joblib
│   └── metrics.json
├── lightgbm/
│   ├── model.joblib
│   └── metrics.json
└── xgboost/
    ├── model.joblib
    └── metrics.json

If a model with the same name already exists, the training pipeline may create a suffixed directory:

decision_tree_1/
random_forest_1/
catboost_1/
Files
model.joblib

Serialized trained model.

This file is loaded during autonomous or suggestion sessions when the selected decider type is:

model
metrics.json

Training metadata and evaluation results.

Typical contents:

{
  "schema": "penhackit.training.v1",
  "trained_at_utc": "2026-05-21T10:30:00Z",
  "dataset_path": "workspace/data/datasets/session_dataset.jsonl",
  "model_type": "decision_tree",
  "n_samples": 120,
  "n_features": 18,
  "feature_names": [
    "goal_type",
    "target_type",
    "focus_level",
    "has_focus_host",
    "has_focus_service",
    "net_ipv4_count",
    "net_gw_count",
    "net_if_count",
    "net_arp_count",
    "net_routes_count",
    "hosts_count",
    "services_count",
    "findings_count",
    "last_action_id",
    "last_action_name",
    "last_rc",
    "last_event_type",
    "step_idx"
  ],
  "label_encoding": {
    "enabled": true,
    "classes": [0, 102, 200, 210, 220, 230, 381, 400]
  },
  "accuracy": 0.75,
  "confusion_matrix": [],
  "classification_report": {}
}

The feature_names field defines the exact feature order used during training. The same vectorization schema must be used during inference.

The label_encoding field is used to map internal model classes back to real PenHackIt action_id values.

Example:

Internal class 0 -> action_id 0
Internal class 1 -> action_id 102
Internal class 2 -> action_id 200

This is especially important for XGBoost, because it expects class labels to be consecutive integers.

Model catalogue

PenHackIt currently uses five tree-based models:

decision_tree
random_forest
catboost
lightgbm
xgboost

These models are selected because the decision problem is based on structured tabular state features.

The input is not raw command output, text logs or the full KB. The input is a compact fixed-size state vector derived from the KB.

## 1. Decision Tree

Identifier: decision_tree

Model: sklearn.tree.DecisionTreeClassifier

Decision Tree is the simplest baseline model.

It learns a hierarchy of decision rules over the state features. Each internal node tests one feature, and each leaf predicts an action class.

Example of learned logic:

if services_count == 0:
    predict PORTSCAN_TOP_TCP
else if has_focus_service == true:
    predict ENUM_SERVICE
else:
    predict SELECT_NEXT_HOST
Technical characteristics
Interpretable.
Fast to train.
Fast to execute.
Works with small datasets.
Can overfit easily.
Useful as a baseline and debugging model.
Allows inspection of feature importance and decision paths.
Role in the project

Decision Tree is mainly used as an interpretable baseline. It helps verify that the state representation contains enough information to choose reasonable actions.

It is not expected to be the strongest model, but it is useful for validating the architecture.


## 2. Random Forest

Identifier: random_forest

Model: sklearn.ensemble.RandomForestClassifier

Random Forest is an ensemble of decision trees trained with bootstrapped samples and random feature selection.

Each tree predicts an action, and the forest combines the predictions by majority vote.

Technical characteristics
More robust than a single decision tree.
Reduces overfitting.
Handles nonlinear feature interactions.
Works well with tabular data.
Provides feature importance.
Less interpretable than a single tree.
Usually a strong classical baseline.
Role in the project

Random Forest is used as a robust baseline for the state_t -> action_id task.

It is useful for comparing whether a simple ensemble improves over a single tree when the dataset grows.


## 3. CatBoost

Identifier: catboost

Model: catboost.CatBoostClassifier

CatBoost is a gradient boosting model based on decision trees.

Boosting builds trees sequentially. Each new tree attempts to correct the errors made by the previous trees.

CatBoost is especially strong on tabular datasets and can handle categorical variables natively. In the current MVP, the state is already numerically encoded before training, but CatBoost remains a strong candidate due to its robustness on structured data.

Technical characteristics
Strong performance on tabular data.
Robust gradient boosting implementation.
Handles nonlinear interactions.
Often performs well with limited feature engineering.
Can handle categorical features natively, although the current MVP uses numeric encoding.
Heavier dependency than scikit-learn models.
Less interpretable than Decision Tree.
Role in the project

CatBoost is one of the main advanced candidates for the decision model.

It is suitable for comparing classical tree ensembles with modern boosting methods.

## 4. LightGBM

Identifier: lightgbm

Model: lightgbm.LGBMClassifier

LightGBM is a gradient boosting framework optimized for speed and efficiency.

It grows trees leaf-wise, which can produce strong models but may require careful parameter tuning, especially with small datasets.

Technical characteristics
Very fast training.
Efficient with larger datasets.
Strong performance on tabular data.
Supports class weighting.
Sensitive to dataset size and split parameters.
May stop early if there are not enough samples to create useful leaves.
Can require tuning of parameters such as num_leaves, min_child_samples and min_data_in_leaf.
Role in the project

LightGBM is used as an advanced boosting comparator.

It is expected to become more useful as the dataset grows. With very small Behavioral Cloning datasets, it may produce weak results or warnings if there are not enough samples per action.

## 5. XGBoost

Identifier: xgboost

Model: xgboost.XGBClassifier

XGBoost is a widely used gradient boosting framework based on decision trees.

It is strong on tabular classification tasks and provides robust regularization mechanisms.

Technical characteristics
Strong tabular performance.
Regularized gradient boosting.
Handles nonlinear decision boundaries.
Often competitive with LightGBM and CatBoost.
Requires labels to be encoded as consecutive class IDs.
Needs label decoding to recover the original PenHackIt action_id.
Label encoding requirement

PenHackIt action IDs are semantic identifiers, not consecutive class labels.

Example:

0, 102, 200, 210, 220, 230, 381, 400

XGBoost expects internal classes like:

0, 1, 2, 3, 4, 5, 6, 7

Therefore, the training pipeline encodes action IDs before training:

action_id 0   -> class 0
action_id 102 -> class 1
action_id 200 -> class 2
action_id 210 -> class 3

During inference, the predicted class must be decoded back into the original action_id.

This mapping is stored in metrics.json under:

"label_encoding": {
  "enabled": true,
  "classes": [0, 102, 200, 210, 220, 230, 381, 400]
}

## Role in the project

XGBoost is included as a strong boosting baseline for structured decision learning.

It should be evaluated once the label encoding and decoding pipeline is correctly applied during both training and inference.

Training pipeline

The training pipeline follows these steps:

1. Load JSONL dataset
2. Extract semantic state and action_id
3. Vectorize state into numeric features
4. Encode action_id labels
5. Split into train/test
6. Train selected model
7. Evaluate predictions
8. Save model.joblib
9. Save metrics.json

The original dataset remains semantic and human-readable.