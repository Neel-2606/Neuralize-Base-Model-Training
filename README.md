<div align="center">
  <h1>🚀 Neuralize: Base Model Training Workshop</h1>
  <p>Welcome to the official repository for the Neuralize Machine Learning Workshop. Here, we will train our baseline models (Regression & Classification) entirely on Google Colab!</p>
</div>

---

## 📥 Step 1: Get the Datasets (Directly in Colab)
Instead of downloading files manually to your computer, we can pull them directly from this GitHub repository straight into your Google Colab environment.

1. Open **[Google Colab](https://colab.research.google.com/)** and click **New Notebook**.
2. Copy the code below, paste it into your first Colab cell, and press `Shift + Enter` to run it. 
3. This will instantly download both datasets into your Colab environment!

```bash
!wget https://raw.githubusercontent.com/Neel-2606/Neuralize-Base-Model-Training/main/student_study_hours.csv
!wget https://raw.githubusercontent.com/Neel-2606/Neuralize-Base-Model-Training/main/food_delivery_status.csv
```

*(You can verify they downloaded by clicking the **Folder icon 📁** on the left menu in Colab).*

</br>
</br>
</br>
</br>
</br>



## 📈 Step 2: Train the Regression Model (Predicting Study Hours)
Copy the code below by hovering over it and clicking the **Copy icon** in the top right corner. Paste it into your first Colab cell and press `Shift + Enter` to run it!

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("--- Training Regression Model (Student Study Hours) ---")

# 1. Load Data from Colab
df = pd.read_csv('student_study_hours.csv')

# 2. Split Features (X) and Target (y)
X = df.drop('Additional_Study_Hours_Required', axis=1)
y = df['Additional_Study_Hours_Required']

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Preprocessing Pipeline
numeric_features = ['Current_CGPA', 'Attendance_Percentage', 'Weekly_Self_Study_Hours', 
                    'Coding_Practice_Hours_Per_Week', 'Previous_Semester_SPI', 
                    'Internal_Assessment_Marks', 'Practical_Assignment_Submission']
categorical_features = ['Target_Grade']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Train the Model
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
print(f"R2 Score: {r2:.4f}")

# 7. Save the Model
joblib.dump(model, 'student_hours_model.pkl')
print("Saved model to 'student_hours_model.pkl'")
```

### Understanding the Results:
*   **MAE:** How far off our guess is on average. A MAE of ~4 hours is highly accurate!
*   **R² Score:** Think of this as a grade from 0 to 1. A score of 0.92 means our input features successfully explain 92% of the variance in the study hours. The model found the pattern!

---

</br>
</br>
</br>
</br>
</br>


## 🍕 Step 3: Train the Classification Model (Food Delivery)
Create a **New Code Cell** below the previous one. Copy, paste, and run this code to train the Classification model!

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- Training Classification Model (Food Delivery Status) ---")

# 1. Load Data from Colab
df = pd.read_csv('food_delivery_status.csv') 

# 2. Split Features (X) and Target (y)
X = df.drop('Delivery_Status', axis=1)
y = df['Delivery_Status']

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Preprocessing Pipeline
numeric_features = ['Delivery_Distance_km', 'Delivery_Partner_Rating', 
                    'Restaurant_Preparation_Time_min', 'Order_Value_INR']
categorical_features = ['Weather_Condition', 'Traffic_Level', 'Order_Size', 'Peak_Hour', 'Day_of_Week']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 5. Train the Model
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the Model
joblib.dump(model, 'food_delivery_model.pkl')
print("Saved model to 'food_delivery_model.pkl'")
```

### Understanding the Results:
*   **Accuracy:** Expect around ~77%. This is a realistic score for complex human logistical data!
*   **Confusion Matrix:** Shows exactly where the model got confused (e.g., predicting "Delayed" when it was actually "On Time").

---

</br>
</br>
</br>
</br>
</br>


## 🎨 Step 4: AI Studio Frontend Generation (On Your Laptop)
Now we are going to build a completely custom Web App interface. 

1. Create a new folder on your laptop's desktop and open it in **VS Code**.
2. Go to **Google AI Studio** and paste the massive prompt provided by your instructor. *(Instructors: The prompt is too large for this README, please provide the PTCF prompt directly to the students via chat/discord)*. 
3. **IMPORTANT:** Change the `[STUDENTS TYPE THEIR DESIRED STYLE HERE]` bracket in the prompt to your favorite aesthetic (e.g., "Cyberpunk", "Minimalist Apple", "Retro 80s Neon").
4. AI Studio will generate three files. Save them inside your VS Code folder as:
   *   `index.html`
   *   `style.css`
   *   `script.js`

---

</br>
</br>
</br>
</br>
</br>


## 🔌 Step 5: Start the API Server in Colab
Your frontend needs a backend API to talk to. We will run a Flask server inside Colab and use Localtunnel to expose it to the internet!

Create a new cell in your Colab notebook, paste the code below, and run it.

```python
!pip install flask flask-cors
!npm install -g localtunnel

import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Load our saved models
reg_model = joblib.load('student_hours_model.pkl')
clf_model = joblib.load('food_delivery_model.pkl')

app = Flask(__name__)
CORS(app) # Enables frontend to talk to backend

@app.route('/predict-study', methods=['POST'])
def predict_study():
    data = request.json
    df = pd.DataFrame([data])
    pred = reg_model.predict(df)[0]
    return jsonify({"prediction": float(pred)})

@app.route('/predict-delivery', methods=['POST'])
def predict_delivery():
    data = request.json
    df = pd.DataFrame([data])
    pred = clf_model.predict(df)[0]
    return jsonify({"prediction": str(pred)})

# Start Flask in the background
def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_flask).start()

# Start Localtunnel to get a public URL
print("\n--- YOUR LOCALTUNNEL URL WILL APPEAR BELOW ---\n")
!lt --port 5000
```
When you run this cell, it will print a URL that looks something like `https://funny-raccoon-99.loca.lt`. **Keep this cell running!**

---

</br>
</br>
</br>
</br>
</br>

## 🚀 Step 6: Connect and Run!

1. Copy the `loca.lt` URL generated in Colab.
2. Go to your VS Code on your laptop and open `script.js`.
3. Paste the URL into the `RAW_COLAB_URL` variable at the top of the file:
   `const RAW_COLAB_URL = "https://funny-raccoon-99.loca.lt";`
4. Save the file.
5. Double-click your `index.html` file to open it in your web browser. 
6. Type some inputs into your custom-styled frontend, hit predict, and watch it talk to your Colab ML models in real-time!

