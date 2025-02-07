# 🍽️ AI-Based Food Recognition & Calorie Detection

## 📌 Project Overview
This project is an **AI-powered Food Recognition and Calorie Detection System** using **YOLO (You Only Look Once) object detection model** and nutritional databases. The system processes food images, identifies the food items, and retrieves nutritional values such as **calories, protein, carbohydrates, and fat** from OpenFoodFacts and USDA databases.

### 🔥 Key Features:
✅ **Food Detection** using YOLOv8 model  
✅ **Nutritional Information Retrieval** from OpenFoodFacts & USDA API  
✅ **FastAPI-based API** for image upload and food recognition  
✅ **Auto-Download** of required models and databases  
✅ **Cloud Ready** - Can be deployed using Docker, AWS, or GCP  
✅ **Scalable & Extensible** - Supports additional food datasets

---

## 🗂️ Dataset Details
This project leverages:
- **Food Images:** Used for training and inference with YOLOv8.
- **OpenFoodFacts Database:** Provides food nutrition data.
- **USDA API:** A fallback source for food nutrition details.

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-repo/food-recognition.git
cd food-recognition
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.8+** installed, then run:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the API Server
```sh
python main.py
```
The server will start at `http://0.0.0.0:8000`

---

## 🚀 API Endpoints
### 📤 Upload Food Image
**Endpoint:** `/upload/`  
**Method:** `POST`  
**Description:** Uploads an image, detects food items, and retrieves nutrition info.

#### Example Request:
```sh
curl -X 'POST' \
  'http://0.0.0.0:8000/upload/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@food_image.jpg'
```

#### Example Response:
```json
{
  "foods_detected": ["Pizza", "Burger"],
  "nutrition_info": {
    "Pizza": {
      "calories": 285,
      "protein": 12,
      "carbs": 36,
      "fat": 10
    },
    "Burger": {
      "calories": 295,
      "protein": 17,
      "carbs": 33,
      "fat": 14
    }
  }
}
```

---

## 🛠️ Technical Details
### 🔹 Technologies Used:
- **Python 3.8+**
- **YOLOv8 Object Detection**
- **FastAPI** (for building the API)
- **Pandas** (for handling food databases)
- **OpenCV & Pillow** (for image processing)
- **Requests** (for API calls to USDA)

### 🔹 Model:
- YOLOv8 is automatically downloaded and loaded when running the application.

### 🔹 Nutrition Database:
- The OpenFoodFacts CSV is downloaded and preprocessed into a lightweight format.
- If the food is not found in OpenFoodFacts, the system queries the USDA API.

---

## 🏗️ Future Enhancements
🚀 **Improve Food Detection Accuracy** - Fine-tune YOLO model with more food images.  
🚀 **Mobile App Integration** - Develop an Android/iOS app for real-time food scanning.  
🚀 **Dietary Recommendations** - Suggest meal plans based on detected food.  
🚀 **Multi-Language Support** - Expand the system to support multiple languages.

---

## 🤝 Contributing
We welcome contributions! Feel free to submit **issues, feature requests, or pull requests**. Follow the steps below:
1. Fork the repository 🍴
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Added new feature"`)
4. Push to your branch (`git push origin feature-branch`)
5. Open a Pull Request 🚀

---

## 📄 License
This project is licensed under the **MIT License** - You are free to modify and distribute it with attribution. See the `LICENSE` file for details.

---

## 📬 Contact & Support
💬 **Author:** KYK  
📧 **Email:** kyk19301@gmail.com
🌐 **GitHub:** [[your-repo-link](https://github.com/kyk-131/AI-Based-Food-Recognition-and-Calorie-Estimator/tree/main)]

---

### 🎉 Happy Food Tracking & Stay Healthy! 🍏🥦🍕

