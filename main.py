from fastapi import FastAPI, UploadFile, File
from uuid import uuid4
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FitPlate API çalışıyor!"}

model = YOLO("food_best.pt")
results_store = {}

# Besin değerleri sözlüğü (100 gram)
nutrition_info = {
    "bakla": {"kalori": 47, "yağ": 1.0, "protein": 3.6, "karbonhidrat": 4.4},
    "baklava": {"kalori": 428, "yağ": 29.03, "protein": 6.7, "karbonhidrat": 37.62},
    "cikolatali pasta": {"kalori": 371, "yağ": 16.0, "protein": 4.0, "karbonhidrat": 53.0},
    "donut": {"kalori": 452, "yağ": 25.0, "protein": 4.9, "karbonhidrat": 51.0},
    "et": {"kalori": 250, "yağ": 15.0, "protein": 26.0, "karbonhidrat": 0.0},
    "hamburger": {"kalori": 165, "yağ": 6.12, "protein": 7.02, "karbonhidrat": 19.88},
    "haslanmis yumurta": {"kalori": 155, "yağ": 10.61, "protein": 12.58, "karbonhidrat": 1.12},
    "havuclu kek": {"kalori": 389, "yağ": 17.0, "protein": 4.0, "karbonhidrat": 56.0},
    "kapkek": {"kalori": 389, "yağ": 17.0, "protein": 4.0, "karbonhidrat": 56.0},
    "kore mantisi": {"kalori": 170, "yağ": 3.5, "protein": 4.1, "karbonhidrat": 29.7},
    "midye": {"kalori": 86, "yağ": 0.96, "protein": 14.67, "karbonhidrat": 3.57},
    "omlet": {"kalori": 154, "yağ": 11.0, "protein": 11.0, "karbonhidrat": 1.0},
    "pankek": {"kalori": 227, "yağ": 7.0, "protein": 6.0, "karbonhidrat": 35.0},
    "patates kizartmasi": {"kalori": 312, "yağ": 14.73, "protein": 3.43, "karbonhidrat": 41.44},
    "peynir tabagi": {"kalori": 402, "yağ": 33.0, "protein": 25.0, "karbonhidrat": 1.3},
    "pizza": {"kalori": 265, "yağ": 5.0, "protein": 7.0, "karbonhidrat": 48.0},
    "red velvet": {"kalori": 423, "yağ": 21.0, "protein": 4.0, "karbonhidrat": 55.0},
    "salata": {"kalori": 30, "yağ": 1.0, "protein": 0.9, "karbonhidrat": 4.7},
    "sandvic": {"kalori": 160, "yağ": 6.54, "protein": 7.39, "karbonhidrat": 17.56},
    "sarimsakli ekmek": {"kalori": 350, "yağ": 15.0, "protein": 7.0, "karbonhidrat": 45.0},
    "sogan halkasi": {"kalori": 411, "yağ": 22.0, "protein": 4.0, "karbonhidrat": 49.0},
    "soslu makarna": {"kalori": 89, "yağ": 2.2, "protein": 3.3, "karbonhidrat": 13.71},
    "spagetti": {"kalori": 158, "yağ": 1.0, "protein": 6.0, "karbonhidrat": 31.0},
    "tavuk kanat": {"kalori": 203, "yağ": 13.0, "protein": 19.0, "karbonhidrat": 0.0},
    "waffle": {"kalori": 291, "yağ": 14.1, "protein": 7.9, "karbonhidrat": 32.9}
}

# Arka planda çalışacak tahmin fonksiyonu
def detect_task(image_bytes, task_id):
    start_time = time.time()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    results = model(img)
    elapsed = round(time.time() - start_time, 2)

    print(f"MODEL TAHMİN SÜRESİ: {elapsed:.2f} saniye")

    predictions = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls].lower().strip()
            nutrition = nutrition_info.get(label, "bilgi bulunamadı")

            predictions.append({
                "yemek": label,
                "doğruluk": round(conf, 2),
                "besin değeri": nutrition
            })

    results_store[task_id] = predictions

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    task_id = str(uuid4())
    thread = threading.Thread(target=detect_task, args=(image_bytes, task_id))
    thread.start()
    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id in results_store:
        return {"status": "done", "tahminler": results_store[task_id]}
    else:
        return {"status": "processing"}
