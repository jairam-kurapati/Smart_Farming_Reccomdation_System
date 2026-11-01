# Smart Farming Recommendation System 🌱

A comprehensive, machine learning-powered web application designed to help farmers make data-driven decisions. By analyzing soil metrics (Nitrogen, Phosphorus, Potassium, pH) and environmental conditions (Temperature, Humidity, Rainfall), the system accurately recommends the optimal crop to cultivate for maximum yield.

## 💡 How to Use the Application

Using the Smart Farming Recommendation System is simple and straightforward:

1. **Navigate to the Recommendation Page**: Click on the "Recommendation" link in the top navigation bar.
2. **Enter Soil Metrics**: Input your soil's current N (Nitrogen), P (Phosphorus), K (Potassium), and pH levels. You can obtain these from a standard soil test.
3. **Fetch Live Weather (Optional but Recommended)**: 
   - Instead of manually guessing the temperature, humidity, and rainfall, simply type your City, Town, or Village name into the "Location (City/Village)" box.
   - Click **"Fetch Live Weather"**. The system will automatically find your location and populate the environmental fields with real-time meteorological data!
4. **Select a Machine Learning Model**:
   - Choose from Decision Tree, Random Forest, K-Nearest Neighbors, or Support Vector Machine.
   - **Recommended:** Select **"Ensemble (Majority Vote)"** for the highest accuracy, as it combines the knowledge of all models.
5. **Get Recommendation**: Click **"Predict Crop"**.
6. **Review & Download**: Read your recommended crop, view alternative crops, and check the confidence score. Click **"Download PDF Report"** at the bottom to save your results!

---

## ✨ Detailed Features Breakdown

### 1. Multi-Model Machine Learning Inference
The application doesn't just rely on one algorithm; it trains and evaluates four different algorithms simultaneously:
- **Decision Tree**: Fast and highly interpretable. Great for understanding the logic behind a decision.
- **Random Forest**: An ensemble of decision trees that reduces overfitting and provides high accuracy.
- **K-Nearest Neighbors (KNN)**: Looks at the most similar historical data points (neighbors) to make a prediction.
- **Support Vector Machine (SVM)**: Excellent for finding complex boundaries between different crop requirements.

### 2. Ensemble Prediction Mode (The "Meta-Model")
Instead of forcing you to choose one algorithm, the **Ensemble Mode** asks all four models to make a prediction on your data. It then conducts a "Majority Vote". For example, if 3 models say "Rice" and 1 says "Jute", the system confidently recommends "Rice". This drastically reduces individual model bias and provides the highest possible accuracy.

### 3. Live Weather Integration
Farmers often don't know the exact current humidity or rainfall in millimeters. The application integrates with the **Open-Meteo API** to automatically pull:
- Current Temperature (°C)
- Relative Humidity (%)
- Precipitation/Rainfall (mm)

### 4. Intelligent Geocoding
To make the weather feature work, the system needs GPS coordinates. The built-in geocoding engine accepts almost any location string (e.g., "Gangireddypalem, India").
- It first attempts to resolve the location using **OpenStreetMap (Nominatim)**.
- If the village is too small and Nominatim fails, it automatically falls back to **ArcGIS Geocoding**, which has exceptional coverage of rural and unmapped agricultural zones in India.

### 5. PDF Report Generation
Farmers and agricultural consultants need to save and share results. The frontend utilizes `html2pdf.js` to instantly convert the results dashboard into a cleanly formatted, downloadable PDF document. It includes an automatic scroll-fix to ensure the PDF is never blank.

### 6. High-Performance Architecture (Caching)
- **Backend `@lru_cache`**: Python's Least Recently Used cache stores the coordinates of previously searched villages, eliminating the 1-2 second API delay for repeated searches.
- **Frontend `weatherCache`**: JavaScript caches the weather data for coordinates. If you fetch weather for the same field twice in one session, the second fetch is instantaneous.

### 7. Modern UI/UX
- **Toast Notifications**: Instead of annoying, blocking browser `alert()` popups, the system uses smooth, sliding notifications that appear at the bottom right of the screen.
- **Dynamic Iconography**: The results page displays a specific FontAwesome icon tailored to the recommended crop (e.g., a tree for Mango, a seedling for Rice, an apple for Apple).

---

## 🛠️ Tech Stack

### Backend
- **Python 3**: Core programming language.
- **Flask**: Lightweight web framework for routing and API endpoints.
- **Waitress**: Production-grade WSGI server for robust, concurrent request handling.
- **Scikit-Learn**: Machine learning library used for training and running the models.
- **Pandas & NumPy**: Used for data manipulation, preprocessing, and numerical operations.
- **Requests**: Modern HTTP library for handling external API calls safely and efficiently.

### Frontend
- **HTML5 & Vanilla CSS3**: Semantic structure and responsive styling without heavy CSS frameworks.
- **Vanilla JavaScript (ES6+)**: Handles all frontend logic, DOM manipulation, and asynchronous API calls (`fetch`).
- **html2pdf.js**: Client-side PDF generation.
- **FontAwesome**: Dynamic iconography based on predicted crops.

---

## 🚀 Installation & Setup

1. **Clone or Download the Repository**
   Ensure you have the project files locally on your machine.

2. **Install Dependencies**
   It is highly recommended to use a virtual environment. Install the required Python packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models (Optional)**
   If the pre-trained `.pkl` model files are missing from the `/models` directory, you can generate them by running the training script:
   ```bash
   python train.py
   ```

4. **Start the Application**
   Launch the production Waitress server by running:
   ```bash
   python app.py
   ```

5. **Access the Web App**
   Open your preferred web browser and navigate to:
   [http://localhost:5000](http://localhost:5000)

---

## 📁 Project Structure

```text
Final Project/
│
├── app.py                  # Main Flask application and backend API logic
├── train.py                # Script to preprocess data and train ML models
├── requirements.txt        # Project dependency list
├── README.md               # Project documentation
│
├── models/                 # Pre-trained ML models (.pkl files)
│   ├── random_forest_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── templates/              # HTML templates
│   ├── base.html           # Master layout template
│   ├── index.html          # Main landing page
│   ├── recommend.html      # Prediction form and results page
│   └── about.html          # Project information page
│
└── static/                 # Static assets
    ├── style.css           # Global stylesheets
    └── script.js           # Frontend interactive logic
```
