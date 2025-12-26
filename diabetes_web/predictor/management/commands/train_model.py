from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Command(BaseCommand):
    help = "Train Linear Regression model and save artifacts"

    def handle(self, *args, **options):
        base_dir = Path(__file__).resolve().parents[3]  # project root
        data_path = base_dir / "predictor" / "data" / "cleaned_data.csv"
        model_path = base_dir / "predictor" / "ml" / "linear_model.pkl"
        cols_path = base_dir / "predictor" / "ml" / "feature_cols.pkl"

        df = pd.read_csv(data_path)

        target = "diabetes_risk_score"
        feature_cols = [
            "age",
            "physical_activity_minutes_per_week",
            "diet_score",
            "bmi",
            "systolic_bp",
            "hdl_cholesterol",
            "ldl_cholesterol",
            "triglycerides",
            "glucose_fasting",
            "hba1c",
        ]

        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        joblib.dump(pipeline, model_path)
        joblib.dump(feature_cols, cols_path)

        self.stdout.write(self.style.SUCCESS("✅ Train xong và đã lưu model!"))
        self.stdout.write(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")
        self.stdout.write(f"Saved: {model_path}")
