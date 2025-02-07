import io
import os
import requests
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# ==================== AUTO-DOWNLOAD YOLO MODEL ====================
MODEL_PATH = "models/yolov8_food.pt"
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO food detection model...")
    os.makedirs("models", exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    response = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            f.write(chunk)
    print("YOLO model downloaded!")

# Load YOLO model
model = YOLO(MODEL_PATH)

# ==================== AUTO-DOWNLOAD NUTRITION DATABASE ====================
NUTRITION_CSV = "nutrition_data.csv"
NUTRITION_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv"

if not os.path.exists(NUTRITION_CSV):
    print("Downloading OpenFoodFacts database in chunks...")
    with requests.get(NUTRITION_URL, stream=True) as response:
        response.raise_for_status()
        with open(NUTRITION_CSV, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                file.write(chunk)
    print("Download complete!")

# Read only required columns in chunks
print("Processing OpenFoodFacts database...")
df_iter = pd.read_csv(NUTRITION_CSV, delimiter="\t", usecols=["product_name", "energy-kcal_100g", "proteins_100g", "carbohydrates_100g", "fat_100g"], iterator=True, chunksize=10000)
nutrition_db = pd.concat(df_iter, ignore_index=True)
nutrition_db.to_csv(NUTRITION_CSV, index=False)
print("Database ready!")

# ==================== FASTAPI IMAGE UPLOAD ENDPOINT ====================
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Detects food items in the image and fetches their nutritional info."""
    image = Image.open(io.BytesIO(await file.read()))

    # Run YOLO model
    results = model(image)
    detected_foods = set()

    for result in results:
        for box in result.boxes:
            detected_foods.add(result.names[int(box.cls)])

    if not detected_foods:
        return {"error": "No food detected"}

    # Fetch nutrition info
    nutrition_info = {}
    for food in detected_foods:
        nutrition_info[food] = get_nutrition(food)

    return {"foods_detected": list(detected_foods), "nutrition_info": nutrition_info}


# ==================== NUTRITION LOOKUP FUNCTIONS ====================
def get_nutrition(food_name):
    """Tries OpenFoodFacts first, then USDA API."""
    nutrition = search_local_db(food_name)
    if not nutrition:
        nutrition = fetch_usda(food_name)
    return nutrition if nutrition else {"calories": "Unknown", "protein": "Unknown", "carbs": "Unknown", "fat": "Unknown"}


def search_local_db(food_name):
    """Search OpenFoodFacts CSV for nutrition data."""
    match = nutrition_db[nutrition_db["product_name"].str.contains(food_name, case=False, na=False)]
    if not match.empty:
        row = match.iloc[0]
        return {
            "calories": row["energy-kcal_100g"],
            "protein": row["proteins_100g"],
            "carbs": row["carbohydrates_100g"],
            "fat": row["fat_100g"]
        }
    return None


def fetch_usda(food_name):
    """Fetch nutrition data from USDA API."""
    USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
    USDA_API_KEY = "54tjbbR0SsWeESfETrxtmGRibk0qtO7etLWneN97"  # Replace with your own API key

    try:
        params = {"query": food_name, "api_key": USDA_API_KEY}
        response = requests.get(USDA_API_URL, params=params)
        data = response.json()

        if "foods" in data and data["foods"]:
            nutrients = data["foods"][0]["foodNutrients"]
            return {
                "calories": find_nutrient(nutrients, 208),
                "protein": find_nutrient(nutrients, 203),
                "carbs": find_nutrient(nutrients, 205),
                "fat": find_nutrient(nutrients, 204)
            }
    except Exception:
        pass

    return None


def find_nutrient(nutrients, id):
    """Extract specific nutrient value from USDA response."""
    for nutrient in nutrients:
        if nutrient["nutrientId"] == id:
            return nutrient["value"]
    return "Unknown"


# ==================== RUN FASTAPI SERVER ====================
if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
