from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

# نبدأ التطبيق
app = FastAPI()

# نحمل الموديل
model = YOLO(r'runs/detect/train37/weights/best.pt')  # حط اسم موديلك هنا

# نجهز مسار يستقبل الصور
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)[0]

    # نحسب كم كائن
    classes = results.names
    counts = {}

    for c in results.boxes.cls.cpu().numpy():
        label = classes[int(c)]
        counts[label] = counts.get(label, 0) + 1

    return JSONResponse(content={"results": counts})

