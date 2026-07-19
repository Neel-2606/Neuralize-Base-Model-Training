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

## 🎨 Step 4: AI Studio Frontend Prompt

Once you have generated your `.pkl` files, we are going to build a customized Web App interface using Google AI Studio. 

Copy the prompt below and paste it into AI Studio. **Make sure to change the `[YOUR STYLE HERE]` bracket at the end to your favorite aesthetic (e.g., Cyberpunk, Retro, Minimalist).**

```text
Persona: You are an expert Full-Stack Web Developer with a strong eye for UI/UX design.

Task: Generate the complete frontend code for a Food Delivery Predictor web application. The frontend must collect user inputs via a form and display the prediction returned by my backend API.

Context: I have a backend API running locally that accepts JSON POST requests at `http://localhost:5000/predict`. 
The JSON payload format it expects is exactly: 
{"Distance": 10, "Weather": "Rainy", "Traffic": "High", "OrderSize": "Medium", "Rating": 4.5, "PrepTime": 20, "PeakHour": "Yes", "DayOfWeek": "Friday", "OrderValue": 500}
The API returns a JSON response: {"prediction": "Delayed"} or {"prediction": "On Time"}. 

Format: 
Output exactly three distinct code blocks representing three files in the following strict folder structure. Do not use any external frameworks (no React, no Vue, no Tailwind). Use pure HTML, CSS, and vanilla JS.

1. index.html: Contains a form with logical UI elements (dropdowns for text categories, sliders/number inputs for numeric categories) for the 9 inputs listed above, and a submit button. Ensure all input elements have clear id attributes.
2. script.js: Contains an async function that intercepts the form submission (e.preventDefault()), reads the values using the element IDs, builds the exact JSON payload expected, sends a POST request to http://localhost:5000/predict using the native Fetch API, and dynamically updates a results <div> on the page with the returned prediction text. Handle basic error states (like API being down).
3. style.css: Apply a [STUDENTS TYPE THEIR DESIRED STYLE HERE, e.g., "Cyberpunk", "Minimalist Apple", "Retro 80s Neon"] aesthetic. Be highly creative with the CSS, utilizing modern hover effects, distinct color palettes, glassmorphism, and polished typography. The UI should look premium, responsive, and completely custom to this theme. Do not write generic CSS.
```
