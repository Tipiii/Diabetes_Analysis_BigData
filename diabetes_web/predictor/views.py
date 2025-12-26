from django.shortcuts import render
# Create your views here.
from .forms import PredictForm
from .ml_utils import predict_risk_score

def predict_view(request):
    result = None

    if request.method == "POST":
        form = PredictForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            score = predict_risk_score(data)
            result = round(score, 4)
    else:
        form = PredictForm()

    return render(request, "predict.html", {"form": form, "result": result})

