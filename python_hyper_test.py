from ultralytics import YOLO
from multiprocessing import freeze_support

hp_param = {"epochs": 200,
        "optimizer": "AdamW",
        "batch": 20,
        "lr0": 0.00303,
        "lrf": 0.02187,
        "momentum": 0.85682,
        "weight_decay": 0.00072,
        "warmup_epochs": 3.64799,
        "warmup_momentum": 0.36812,
        "box": 2.07569,
        "cls": 0.43868,
        "dfl": 1.263,
        "hsv_h": 0.01332,
        "hsv_s": 0.75363,
        "hsv_v": 0.3853,
        "degrees": 0.0,
        "translate": 0.04128,
        "scale": 0.55634,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.46024,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0}

if __name__ == '__main__':
    freeze_support()
    model = YOLO("yolov8s.pt")
    model.train(data="data.yaml", **hp_param, name="YOLOv8s_hp_ep200")
    