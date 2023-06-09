from django.db import models
from django.contrib.auth.models import User

class Keypoint(models.Model):
    x = models.FloatField()
    y = models.FloatField()
    score = models.FloatField()

class MotionData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    keypoints = models.ManyToManyField(Keypoint)
