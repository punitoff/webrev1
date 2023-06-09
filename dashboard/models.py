from django.db import models
from django.contrib.auth.models import User
from motion_capture.models import MotionData

class Dashboard(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    motion_data = models.ForeignKey(MotionData, on_delete=models.CASCADE)
    additional_metric = models.IntegerField()