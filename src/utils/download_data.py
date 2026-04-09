from roboflow import Roboflow
import os

save_path = "/media/ml4u/Challenge-4TB/baonhi/my_data"

rf = Roboflow(api_key="C1WFxioKoQo2T2Gmfo1F")
project = rf.workspace("dao-duong").project("my_dataset-8fpvk")
dataset = project.version(1).download("yolov8", location=save_path)

