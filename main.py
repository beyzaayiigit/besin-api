from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FitPlate API çalışıyor!"}

model = YOLO("food_best.pt")

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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    start = time.time()
    results = model(image)
    print(f"MODEL TAHMİN SÜRESİ: {time.time() - start:.2f} saniye")

    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower().strip()
            conf = float(box.conf[0])
            nutrition = nutrition_info.get(label, "bilgi bulunamadı")

            detections.append({
                "yemek": label,
                "doğruluk": round(conf, 2),
                "besin değeri": nutrition
            })

    return {"tahminler": detections}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)