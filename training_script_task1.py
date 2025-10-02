from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
   data='/home/jaeden/buildingNumDataset.yaml',
   imgsz=400,
   epochs=100,
   batch=16,
   name='yolov8s_v8',
   patience=20,
   conf=0.35,
   max_det=10
)


model = YOLO('/root/runs/detect/yolov8s_v88/weights/best.pt')
results = model.val(batch=8, project='/home/jaeden/BuildingNumIdentifier/output', name='val_results') 