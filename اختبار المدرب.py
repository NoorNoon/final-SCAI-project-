from ultralytics import YOLO
import cv2

# تحميل النموذج المدرّب (بدّل بالمسار الصحيح لنموذجك)
model = YOLO(r'runs/detect/train40/weights/best.pt')

# تحميل الصورة اللي تبي تجرب عليها
img_path = r'C:\Users\zsaff\PycharmProjects\PythonProject2\HallCount.v1i.yolov8\train\images\2right_frame_57_jpg.rf.bf442e65ab39c8fb05b9c1df008b9698.jpg'
img = cv2.imread(img_path)

# تشغيل النموذج على الصورة
results = model(img)

# استخراج معلومات من النتائج
for result in results:
    for box in result.boxes:
        # إحداثيات البوكس
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # نسبة الثقة
        confidence = float(box.conf[0])

        # رقم الصنف و اسمه
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        # طباعة التفاصيل
        print(f"Object: {class_name}, confedince: {confidence:.2f}, box: ({x1}, {y1}, {x2}, {y2})")

        # رسم البوكس والنص على الصورة
        # رسم البوكس والنص على الصورة - اللون أزرق الآن
        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # أزرق
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # أزرق

# عرض الصورة بالنتائج
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
