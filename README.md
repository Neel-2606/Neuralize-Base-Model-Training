<div align="center">
  <h1>🚀 Neuralize: Base Model Training Workshop</h1>
  <p>Welcome to the official repository for the Neuralize Machine Learning Workshop. Here, we will train our baseline models (Regression & Classification) entirely on Google Colab, and build a beautiful custom frontend on our local laptops!</p>
</div>

<br><br><br><br><br>

---

## 📥 Step 1: Get the Datasets (Directly in Colab)
Instead of downloading files manually to your computer, we can pull them directly from this GitHub repository straight into your Google Colab environment.

1. Open **[Google Colab](https://colab.research.google.com/)** and click **New Notebook**.
2. Copy the code below, paste it into your first Colab cell, and press `Shift + Enter` to run it. 

```bash
!wget https://raw.githubusercontent.com/Neel-2606/Neuralize-Base-Model-Training/main/student_study_hours.csv
!wget https://raw.githubusercontent.com/Neel-2606/Neuralize-Base-Model-Training/main/food_delivery_status.csv
```

<br><br><br><br><br>

---

## 📈 Step 2: Train the Regression Model (Predicting Study Hours)
Create a new cell in Colab, paste this code, and run it. This trains our study hours model and saves it as a `.pkl` file.

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

<br><br><br><br><br>

---

## 🍕 Step 3: Train the Classification Model (Food Delivery)
Create a new cell in Colab, paste this code, and run it. This trains our delivery model and saves it as a `.pkl` file.

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
---

<br><br><br><br><br>



## 🎨 Step 4: AI Studio Frontend Generation (On Your Laptop)
Now we are going to build a completely custom Web App interface. 

1. Create a new folder on your laptop's desktop and open it in **VS Code**.
2. Go to **Google AI Studio** and paste the massive prompt provided below.
3. **IMPORTANT:** Change the `[STUDENTS TYPE THEIR DESIRED STYLE HERE]` bracket in Part 6 of the prompt to your favorite aesthetic (e.g., "Cyberpunk", "Minimalist Apple", "Retro 80s Neon").

<details>
<summary><b>Click to expand and copy the Full AI Studio Prompt</b></summary>

```text
PERSONA
You are a Staff-level Full-Stack Product Engineer who has shipped production dashboards for AI/ML companies. You write fault-tolerant, accessible, enterprise-grade code, AND you have the design sensibility of a world-class senior product designer specializing in modern enterprise software, with deep expertise in information hierarchy, interaction design, accessibility, typography, and visual systems. You never ship code that looks like a "college project," a Bootstrap admin template, or a default HTML form. Every decision — architecture and visual — must feel deliberate.

GOAL
Generate a complete, production-ready HTML5 + CSS3 + ES6+ JavaScript frontend for "Neuralize AI Prediction Hub" — a dashboard containing two ML inference tools (a classification model and a regression model) connected to a Colab backend via Localtunnel. The result must be robust enough to run without modification (except pasting a URL) AND polished enough to look like a real SaaS product, not a workshop demo.

Note on "production-ready": in this context it means reliable, maintainable, accessible, and easy to understand — not maximally abstracted or feature-complete. This is a three-file vanilla application; avoid introducing unnecessary abstractions, speculative extensibility, extra layers of indirection, or complexity that isn't directly required by the rules in this prompt.

OUTPUT FORMAT & RESTRICTIONS
Output ONLY these three files, in this exact order, each in its own fenced code block:
1. index.html
2. style.css
3. script.js

No introductory text. No closing/summary text. No markdown commentary outside the three code blocks. Do not invent a fourth file, a build step, a framework, or a package.json. Vanilla HTML/CSS/JS only.

PART 1 — MANDATORY ARCHITECTURE RULES (DO NOT VIOLATE)
index.html MUST begin with <!DOCTYPE html> on the very first line, followed by <html lang="en">. This is non-negotiable and easy to check for.
Do NOT rename, alter casing, or restructure any HTML element ID or JSON payload key specified in Part 3 or Part 4 (e.g., Current_CGPA must remain exactly Current_CGPA). You MAY add extra CSS classes, wrapper <div>s, and ARIA attributes freely — but never touch the specified IDs/keys.
Use ONLY these two backend endpoints. Do not invent, rename, or add any others:
POST /predict-delivery
POST /predict-study

Target evergreen modern browsers (latest Chrome, Edge, Firefox). No experimental, deprecated, or vendor-prefixed-only APIs.
Zero console errors and zero uncaught promise rejections during normal operation, including all failure paths described in Part 5.
No inline CSS, no inline JS, no inline HTML event attributes (onclick, onchange, etc.). All styling in style.css, all behavior in script.js, all bindings via addEventListener.
No global variables or functions except RAW_COLAB_URL and COLAB_URL. Every helper function, DOM reference, constant, and event listener must be declared inside a single document.addEventListener("DOMContentLoaded", () => { ... }) block.
style.css linked in <head> via <link rel="stylesheet" href="style.css">.
<meta charset="UTF-8"> and <meta name="viewport" content="width=device-width, initial-scale=1.0"> present in <head>.
script.js included immediately before </body> as <script src="script.js"></script> — do NOT use type="module".
Do not silently swallow errors. Every failure path in Part 5 must render a visible, human-readable message inside the relevant result container — never just console.log and stop.
Avoid overengineering. Do NOT build custom dropdown components, custom scrollbars, or custom checkbox/radio libraries. Use native <select>, <input>, and browser controls, styled entirely via CSS. Prioritize reliability and predictable behavior across all laptops/browsers over visual novelty.

PART 2 — API CONTRACTS (EXPLICIT REQUEST/RESPONSE SCHEMAS)
2.1 Endpoint: POST /predict-delivery
Request body (JSON), keys and types exactly as follows:
{"Delivery_Distance_km": 10.0, "Weather_Condition": "Sunny", "Traffic_Level": "Medium", "Order_Size": "Medium", "Delivery_Partner_Rating": 4.5, "Restaurant_Preparation_Time_min": 20, "Peak_Hour": "No", "Day_of_Week": "Monday", "Order_Value_INR": 500}
Numeric fields sent as JS Number (not strings). String fields sent exactly as selected (case-sensitive, matching dropdown option text exactly).
Expected success response (JSON): { "prediction": "On Time" } or { "prediction": "Delayed" }
Any other value in prediction, or a missing prediction key, must be treated as a schema mismatch — do not assume, coerce, or guess.

2.2 Endpoint: POST /predict-study
Request body (JSON):
{"Current_CGPA": 7.5, "Attendance_Percentage": 80.0, "Weekly_Self_Study_Hours": 20, "Coding_Practice_Hours_Per_Week": 15, "Previous_Semester_SPI": 7.5, "Internal_Assessment_Marks": 20, "Practical_Assignment_Submission": 85, "Target_Grade": "A"}
Expected success response (JSON): { "prediction": 12.8 }
prediction must be a number (or numeric string coercible via Number()); format strictly with Number(data.prediction).toFixed(1) before display. If prediction is missing, null, or NaN after coercion, treat as schema mismatch.

2.3 Shared Request Headers (both endpoints)
Content-Type: application/json
bypass-tunnel-reminder: true

PART 3 — index.html SPECIFICATION
3.1 Structure
Begin with <!DOCTYPE html> and <html lang="en">. Use semantic HTML throughout: <header>, <main>, two <section> cards inside a grid wrapper, <footer>.

3.2 Header
Title: "Neuralize AI Prediction Hub"
Subtitle: "Real-Time Machine Learning Model Inference Dashboard"
Status badge, ID apiStatusIndicator, default text: "Status: Standby"
The badge must support four visual states (Standby, Connecting, Connected, Error) that transition smoothly via CSS transition on color/background properties only — never via abrupt flashing or a hard cut.

3.3 Global Layout
Responsive split grid, two cards. Below 900px: stack vertically. Fully usable down to 320px width, no horizontal scroll at any breakpoint.
The layout must remain visually stable while the browser window is actively resized — no overlapping elements, no clipped gradients, no shadow overflow/cutoff, at any width between 320px and large desktop.

3.4 Accessibility (mandatory, non-negotiable)
Every input/select has a <label for="..."> matching its id.
Every submit button has a descriptive aria-label.
deliveryResult and studyResult both have aria-live="polite".
Focus order must be logical (top-to-bottom, left-to-right per card); no positive tabindex values.
All interactive elements reachable and operable via keyboard alone.
All tappable elements (buttons, selects, inputs) must have a minimum touch target of 44×44px on mobile viewports, per Apple's Human Interface Guidelines — pad hit areas with CSS if the visible element is smaller.

3.5 LEFT CARD — Food Delivery Delay Predictor
Card title: "🍔 Food Delivery Delay Predictor"
Small muted subtitle under title: "Classification Model"
Form ID: deliveryForm
Result container ID: deliveryResult, initial placeholder: "Fill out inputs and click Predict."
Fields:
1. Delivery_Distance_km (number) min 1.0, max 25.0, step 0.1, Default 10
2. Weather_Condition (select) Sunny, Cloudy, Rainy, Stormy
3. Traffic_Level (select) Low, Medium, High
4. Order_Size (select) Small, Medium, Large
5. Delivery_Partner_Rating (number) min 1.0, max 5.0, step 0.1, Default 4.5
6. Restaurant_Preparation_Time_min (number) min 5, max 45, step 1, Default 20
7. Peak_Hour (select) Yes, No
8. Day_of_Week (select) Monday...Sunday
9. Order_Value_INR (number) min 100, max 2500, step 10, Default 500
Submit button ID: deliverySubmitBtn, text "Predict Delivery Status", aria-label="Predict Food Delivery Status".

3.6 RIGHT CARD — Student Study Hours Predictor
Card title: "📚 Student Study Hours Predictor"
Small muted subtitle under title: "Regression Model"
Form ID: studyForm
Result container ID: studyResult, initial placeholder: "Fill out inputs and click Calculate."
Fields:
1. Current_CGPA (number) min 5.0, max 10.0, step 0.01, Default 7.5
2. Attendance_Percentage (number) min 50.0, max 100.0, step 0.1, Default 80.0
3. Weekly_Self_Study_Hours (number) min 0, max 60, step 0.5, Default 20
4. Coding_Practice_Hours_Per_Week (number) min 0, max 35, step 0.5, Default 15
5. Previous_Semester_SPI (number) min 5.0, max 10.0, step 0.01, Default 7.5
6. Internal_Assessment_Marks (number) min 0, max 30, step 0.5, Default 20
7. Practical_Assignment_Submission (number) min 0, max 100, step 1, Default 85
8. Target_Grade (select) O, A+, A, B+, B, C
Submit button ID: studySubmitBtn, text "Calculate Required Hours", aria-label="Calculate Required Study Hours".

PART 4 — script.js SPECIFICATION
4.1 Top-of-file globals (the ONLY allowed globals)
const RAW_COLAB_URL = "YOUR_LOCALTUNNEL_URL_HERE";
const COLAB_URL = RAW_COLAB_URL.replace(/\/+$/, "");

4.2 Everything else lives inside:
document.addEventListener("DOMContentLoaded", () => { // all helpers, constants, listeners here });

4.3 Code Organization
Organize the code inside the DOMContentLoaded callback into clearly commented sections, in this order:
Configuration — DOM element references, constants.
Helpers — updateApiStatus, spinner creation/removal, generic small utilities.
Validation — validateInputs and any field-level validation helpers.
API — makeApiRequest and endpoint-specific request builders.
Rendering — functions that write success/error/empty/skeleton states into the result containers.
Event Listeners — the two form submit listeners, wired last, calling into the sections above.
Use a // ---- SECTION NAME ---- comment banner above each section so the file reads top-to-bottom as a clear pipeline.

4.4 Required Helper Functions
a) updateApiStatus(state)
"connecting" → text "Status: Connecting…", amber styling class.
"connected" → text "Status: Connected", green styling class.
"error" → text "Status: Error", red styling class.
Toggle via CSS classes only (no inline styles); rely on CSS transitions for the visual change, never abrupt.

b) Spinner / Button State Handler
Build the spinner element dynamically in JS (document.createElement), never hardcoded in HTML.
On submit start: disable button, insert spinner inline next to label. The button's rendered width and height must stay identical across normal, loading, and disabled states — reserve space with CSS (e.g. fixed min-width/height, flexbox centering) so nothing shifts.
On finish (success or failure): remove spinner, re-enable button. This cleanup MUST live in the handler's finally block, not inside the fetch wrapper.

c) validateInputs(elementIds)
For each ID: read the element, parse with parseFloat/Number as appropriate for numeric fields.
Reject if: value is empty, isNaN(value), !isFinite(value), or outside the element's own min/max (read Number(input.min) and Number(input.max), never compare raw strings).
For select fields: reject only if value is empty/unselected.
On failure: for every invalid field, add an "invalid" CSS class that gives it an accent-colored border and triggers a small inline helper message beneath that specific field (e.g. "Enter a value between 1.0 and 25.0") — without changing the field's layout position or size. Do not rely solely on one generic global error message when the cause is field-level; show the user exactly which fields need fixing, in place.
Return a boolean or a list of invalid field IDs.

d) makeApiRequest(url, payload) — unified fetch wrapper
Create AbortController, timeout via setTimeout(() => controller.abort(), 12000).
fetch(url, { method: "POST", headers: {"Content-Type": "application/json", "bypass-tunnel-reminder": "true"}, body: JSON.stringify(payload), signal: controller.signal }).
Always clearTimeout(timeoutId) before returning or throwing.
Check response.headers.get("content-type") includes "application/json" BEFORE calling .json(). If not → throw new Error("LOCAL_TUNNEL_SPLASH").
If !response.ok → throw new Error("HTTP_" + response.status).
Otherwise return parsed JSON.

4.5 Required Failure Handling (must never throw uncaught)
Condition | User-facing message
COLAB_URL still equals placeholder | "Please paste your active Localtunnel URL in script.js"
Client-side validation fails | Field-level messages per input (see 4.4c), plus a brief summary line: "Please correct the highlighted fields."
error.name === "AbortError" | "Connection timed out after 12 seconds. Please verify your Localtunnel URL or Colab runtime."
error.message === "LOCAL_TUNNEL_SPLASH" | "Localtunnel URL has not been authorized. Please open the URL directly in your browser once to accept the reminder screen."
error.message starts with "HTTP_" | Calm, specific message reflecting the status code, e.g. "The server returned an error (HTTP 500). Please check your Colab backend logs."
Response JSON missing/invalid prediction key | "Invalid schema format received from server."
Any other unexpected error | Generic fallback: "Something went wrong. Please try again."
Each error state must call updateApiStatus("error"). Successful responses must call updateApiStatus("connected"). The moment a request is fired, call updateApiStatus("connecting").

4.6 Form Handler Flow (both forms)
e.preventDefault() immediately.
Run validateInputs([...]). If invalid → mark invalid fields per 4.4c, show summary line, return early (no spinner/network call).
If COLAB_URL is still the placeholder → show placeholder message, return early.
Set button to loading state (preserving width/height), call updateApiStatus("connecting"), and render a skeleton placeholder into the result container (see Part 5.8).
try { const data = await makeApiRequest(...); ...render success... } catch (error) { ...map error to message per table above... } finally { ...remove spinner, re-enable button... }
Delivery success → check data.prediction === "On Time" or "Delayed" explicitly; anything else → schema mismatch message.
Study success → coerce Number(data.prediction); if isNaN → schema mismatch message; else format .toFixed(1) and render "Additional Study Required: X.X Hours/Whole Reading vacation".

4.7 State Management Discipline
No shared mutable state between the two forms — each handler manages its own button/spinner/result container independently.
Do not use global flags; keep loading state scoped to each handler's closure.

PART 5 — style.css SPECIFICATION (PREMIUM, RESTRAINED SAAS DESIGN)
5.1 Design Philosophy (read before writing any CSS)
Create an interface that feels like a premium AI SaaS product designed by an experienced product design team. The user should be able to immediately understand — without reading any documentation — what each predictor does, where to interact, what is currently loading, what succeeded, and what failed.
The design should emphasize: Exceptional visual hierarchy, Consistent spacing, Excellent typography, Clear information architecture, Minimal cognitive load, High readability, Purposeful motion, Restrained use of color, Strong accessibility, Professional polish.
The interface should achieve the level of polish expected from a modern enterprise AI SaaS application. Do not imitate or reproduce the branding, layouts, visual assets, or distinctive design language of any specific company or product — the goal is matching a level of restraint and polish, not copying anyone's specific look. You have full creative freedom over the specific color palette, theme (light or dark), and visual motifs, as long as every choice serves the philosophy above.

5.2 Visual Identity — Freedom Within the Philosophy
Since the two cards represent two distinct ML tools, consider giving each a subtle, distinct accent identity (e.g. a different accent hue per card) so users can tell them apart at a glance — but keep both accents restrained and harmonious with the overall palette, not competing or garish.
Typography and font-loading fallback strategy (important for offline/restricted workshop networks):
You may load a font via Google Fonts <link> when internet access is available. Always define the CSS font-family stack with graceful system fallbacks so the page still looks intentional with zero internet access, e.g.:
font-family: "YourChosenFont", system-ui, -apple-system, "Segoe UI", sans-serif;
Never let a failed font load result in a default serif fallback.

5.3 CSS Organization
Organize the stylesheet into clearly labeled sections: Variables, Reset, Layout, Typography, Header, Cards, Forms, Buttons, States, Animations, Responsive.

5.4 Design Tokens
Define all tokens once in :root. Use them everywhere. No magic numbers or ad-hoc hex values inside component rules.

5.5 Layout
Max-width container (~1280px), centered, responsive outer padding.
Card grid: grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 28px; collapsing to single column under 900px.

5.6 Form Fields
Label above input, small/uppercase-tracked, muted color.
Inputs/selects: dark translucent fill, thin border, comfortable height (~44–48px), rounded corners.
Helper text under numeric fields showing valid range (e.g. "Range: 1.0–25.0 km").
:focus-visible: accent-colored border + a single soft focus ring — smooth --transition-fast.

5.7 Buttons
Full width within card, clear label, subtle accent-gradient background matching the module.
Fixed dimensions: width and height must be identical across normal, loading, and disabled states.

5.8 Distinct State Treatments (empty / loading / success / error)
Empty/placeholder state: Include a small, quiet illustration, a short placeholder heading, and one line of helper description below it.
Loading state: Render a skeleton loader inside the result container whose size and shape closely match the eventual success panel.
Success state: Animate in using opacity (0→1), translateY (e.g. 8px→0), and scale (0.98→1) simultaneously, over ~220ms with a single ease-out curve.

5.10 Responsive & Stability Requirements
No horizontal scroll at 320px, 375px, 768px, 900px, or 1440px.

PART 6 — STYLE INSTRUCTION FOR STUDENT
Apply a [STUDENTS TYPE THEIR DESIRED STYLE HERE, e.g., "Cyberpunk", "Minimalist Apple", "Retro 80s Neon"] aesthetic. Be highly creative with the CSS, utilizing modern hover effects, distinct color palettes, glassmorphism, and polished typography. The UI should look premium, responsive, and completely custom to this theme. Do not write generic CSS.
```
</details>

4. AI Studio will generate three files. Save them inside your VS Code folder as:
   *   `index.html`
   *   `style.css`
   *   `script.js`

<br><br><br><br><br>

---

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

<br><br><br><br><br>

---

## 🚀 Step 6: Connect and Run!

1. Copy the `loca.lt` URL generated in Colab.
2. Go to your VS Code on your laptop and open `script.js`.
3. Paste the URL into the `RAW_COLAB_URL` variable at the top of the file:
   `const RAW_COLAB_URL = "https://funny-raccoon-99.loca.lt";`
4. Save the file.
5. Double-click your `index.html` file to open it in your web browser. 
6. Type some inputs into your custom-styled frontend, hit predict, and watch it talk to your Colab ML models in real-time!
