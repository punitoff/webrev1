# Generated by Django 4.2.2 on 2023-06-07 08:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("motion_capture", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Keypoint",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("x", models.FloatField()),
                ("y", models.FloatField()),
                ("score", models.FloatField()),
            ],
        ),
        migrations.RemoveField(
            model_name="motiondata",
            name="data",
        ),
        migrations.AddField(
            model_name="motiondata",
            name="keypoints",
            field=models.ManyToManyField(to="motion_capture.keypoint"),
        ),
    ]
