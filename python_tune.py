from ultralytics import YOLO


path_to_project = "runs\\detect\\all_tune"

model = YOLO("yolov8s.pt")
model.tune(data="data.yaml", epochs=30, iterations=30, optimizer="AdamW", plots=False, save=False, val=False, batch=-1, project=path_to_project, name="tune_yolov8s")