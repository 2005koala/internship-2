from ultralytics import YOLO

# ����YOLOv8��COCOԤѵ��ģ��
model = YOLO('yolov8n.yaml')  # 'yolov8n' ��һ��nano�����ģ�ͣ��ʺϿ��ٲ��ԣ���Ҳ����ѡ�������� 'yolov8s', 'yolov8m', 'yolov8l', �� 'yolov8x'

# �������ֱ�Ӽ��عٷ��ṩ��Ԥѵ��Ȩ�أ�����������
model = YOLO('yolov8n.pt')  # �����������Ҫ���ص���nano�汾��Ԥѵ��ģ��

# ��һ��ͼƬ����Ԥ��
results = model.predict(source="https://ultralytics.com/images/bus.jpg")

# ��ӡ���
for r in results:
    print(r.boxes)  # ����������Ϣ
