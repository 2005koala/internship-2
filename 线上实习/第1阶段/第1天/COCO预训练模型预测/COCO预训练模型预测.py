from ultralytics import YOLO

# 加载YOLOv8的COCO预训练模型
model = YOLO('yolov8n.yaml')  # 'yolov8n' 是一个nano级别的模型，适合快速测试；你也可以选择其他如 'yolov8s', 'yolov8m', 'yolov8l', 或 'yolov8x'

# 如果你想直接加载官方提供的预训练权重，可以这样：
model = YOLO('yolov8n.pt')  # 这里假设你想要加载的是nano版本的预训练模型

# 对一张图片进行预测
results = model.predict(source="https://ultralytics.com/images/bus.jpg")

# 打印结果
for r in results:
    print(r.boxes)  # 输出检测框的信息
