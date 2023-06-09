from .models import Dashboard
from django.shortcuts import render

def dashboard(request):
    dashboards = Dashboard.objects.all()
    context = {
        'dashboards': dashboards,
        'lines': [
            "Welcome to my Real-time Webcam Motion Capture demoapp!",
            "Track landmarks of hands and face using the Holistic model from Mediapipe.",
            "This is a demo of how AI and computer vision can be used for your possible project.",
            "Move your hand or nod your head, and it'll draw a box around that for 2 seconds.",
            "Click below to start your experience."
        ]
    }
    return render(request, 'dashboard/dashboard.html', context)


