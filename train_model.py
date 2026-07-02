import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = "/content/drive/MyDrive"

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "transactions.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("/content/drive/MyDrive/transactions.csv")

print(df.head())
print()

print(df.info())
print()

# ==========================================================
# Features
# ==========================================================

TARGET = "Fraud"

X = df.drop(TARGET, axis=1)

y = df[TARGET]


# ==========================================================
# Feature Types
# ==========================================================

categorical_features = [

    "Transaction_Type",
    "Merchant_Category",
    "Device_Type"

]

numerical_features = [

    "Amount",
    "Transaction_Hour",
    "Customer_Age",
    "Previous_Fraud"

]


# ==========================================================
# Preprocessor
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            StandardScaler(),

            numerical_features

        ),

        (

            "cat",

            OneHotEncoder(handle_unknown="ignore"),

            categorical_features

        )

    ]

)


# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ==========================================================
# Fit Preprocessor
# ==========================================================

print("\nFitting Preprocessor...\n")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ==========================================================
# Train Model
# ==========================================================

print("Training Random Forest...\n")

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    max_depth=15,

    class_weight="balanced"

)

model.fit(X_train_processed, y_train)

print("Training Completed.\n")


# ==========================================================
# Evaluation
# ==========================================================

predictions = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, predictions)

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")

print()

print(classification_report(y_test, predictions))

print()

print(confusion_matrix(y_test, predictions))


# ==========================================================
# Save Model
# ==========================================================

joblib.dump(

    model,

    os.path.join(
        MODEL_DIR,
        "fraud_model.pkl"
    )

)

print("fraud_model.pkl saved.")


# ==========================================================
# Save Preprocessor
# ==========================================================

joblib.dump(

    preprocessor,

    os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )

)

print("preprocessor.pkl saved.")


# ==========================================================
# Save Feature Names
# ==========================================================

feature_columns = preprocessor.get_feature_names_out()

joblib.dump(

    feature_columns,

    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )

)

print("feature_columns.pkl saved.")


# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.DataFrame({

    "Feature": feature_columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 15 Important Features\n")

print(importance.head(15))


importance.to_csv(

    os.path.join(
        MODEL_DIR,
        "feature_importance.csv"
    ),

    index=False

)


print("\nFeature Importance CSV saved.")


# ==========================================================
# Finish
# ==========================================================

print("\n" + "=" * 60)
print("Training Completed Successfully")
print("=" * 60)

print("\nSaved Files")

print("-------------------------")

print("fraud_model.pkl")

print("preprocessor.pkl")

print("feature_columns.pkl")

print("feature_importance.csv")

print("\nReady for Prediction.")