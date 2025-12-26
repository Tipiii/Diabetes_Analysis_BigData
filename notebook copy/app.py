import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel

st.title("Diabetes Risk Score Predictor (Spark Linear Regression)")

# 1) Spark session
spark = SparkSession.builder.appName("DiabetesApp").master("local[*]").getOrCreate()

# 2) Khai báo features giống lúc train
features = [
    "age",
    "physical_activity_minutes_per_week",
    "diet_score",
    "family_history_diabetes",
    "bmi",
    "systolic_bp",
    "hdl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "hba1c"
]

# 3) Load model đã lưu (bạn cần save trước)
MODEL_PATH = "models/lr_diabetes_model"   # đổi đúng đường dẫn bạn lưu
lr_model = LinearRegressionModel.load(MODEL_PATH)

# 4) Assembler giống lúc train
assembler = VectorAssembler(inputCols=features, outputCol="features")

# UI inputs
age = st.number_input("age", min_value=1, max_value=120, value=50)
pa = st.number_input("physical_activity_minutes_per_week", min_value=0, value=120)
diet = st.number_input("diet_score", min_value=0.0, max_value=10.0, value=6.5)
family = st.selectbox("family_history_diabetes", [0, 1])
bmi = st.number_input("bmi", min_value=10.0, max_value=60.0, value=27.0)
sysbp = st.number_input("systolic_bp", min_value=70, max_value=250, value=130)
hdl = st.number_input("hdl_cholesterol", min_value=1.0, max_value=150.0, value=45.0)
tg = st.number_input("triglycerides", min_value=1.0, max_value=800.0, value=160.0)
glu = st.number_input("glucose_fasting", min_value=30.0, max_value=300.0, value=115.0)
hba1c = st.number_input("hba1c", min_value=2.0, max_value=20.0, value=6.8)

if st.button("Predict"):
    vals = {
        "age": float(age),
        "physical_activity_minutes_per_week": float(pa),
        "diet_score": float(diet),
        "family_history_diabetes": float(family),
        "bmi": float(bmi),
        "systolic_bp": float(sysbp),
        "hdl_cholesterol": float(hdl),
        "triglycerides": float(tg),
        "glucose_fasting": float(glu),
        "hba1c": float(hba1c),
    }

    sample_df = spark.createDataFrame([Row(**vals)])
    for f in features:
        sample_df = sample_df.withColumn(f, col(f).cast("double"))

    sample_assembled = assembler.transform(sample_df)
    pred = lr_model.transform(sample_assembled).select("prediction").first()["prediction"]

    st.success(f"Predicted diabetes_risk_score = {float(pred):.2f}")
