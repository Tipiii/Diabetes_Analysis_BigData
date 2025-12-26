from django import forms

class PredictForm(forms.Form):
    age = forms.IntegerField(min_value=1, max_value=120, label="Tuổi")
    physical_activity_minutes_per_week = forms.IntegerField(min_value=0, max_value=10000, label="Phút vận động/tuần")
    diet_score = forms.FloatField(min_value=0, max_value=10, label="Điểm chế độ ăn (diet_score)")
    bmi = forms.FloatField(min_value=10, max_value=70, label="BMI")
    systolic_bp = forms.IntegerField(min_value=70, max_value=250, label="Huyết áp tâm thu (systolic)")
    hdl_cholesterol = forms.FloatField(min_value=0, max_value=200, label="HDL cholesterol")
    ldl_cholesterol = forms.FloatField(min_value=0, max_value=400, label="LDL cholesterol")
    triglycerides = forms.FloatField(min_value=0, max_value=1000, label="Triglycerides")
    glucose_fasting = forms.FloatField(min_value=40, max_value=400, label="Glucose lúc đói")
    hba1c = forms.FloatField(min_value=3, max_value=20, label="HbA1c")
