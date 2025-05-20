from flask import Flask, render_template, request, redirect ,jsonify
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/day')
def day():
    return render_template('day.html')

@app.route('/toddlers')
def toddlers():
    return render_template('toddlers.html')

@app.route('/children')
def children():
    return render_template('children.html')

@app.route('/adults')
def adults():
    return render_template('adults.html')

@app.route('/tgame1')
def tgame1():
    return render_template('tgame1.html')

@app.route('/tgame2')
def tgame2():
    return render_template('tgame2.html')

@app.route('/tgame3')
def tgame3():
    return render_template('tgame3.html')

@app.route('/tgame4')
def tgame4():
    return render_template('tgame4.html')

@app.route('/tgame5')
def tgame5():
    return render_template('tgame5.html')

@app.route('/tgame6')
def tgame6():
    return render_template('tgame6.html')

@app.route('/cgame1')
def cgame1():
    return render_template('cgame1.html')

@app.route('/cgame2')
def cgame2():
    return render_template('cgame2.html')

@app.route('/cgame3')
def cgame3():
    return render_template('cgame3.html')

@app.route('/cgame4')
def cgame4():
    return render_template('cgame4.html')

@app.route('/cgame5')
def cgame5():
    return render_template('cgame5.html')

@app.route('/cgame6')
def cgame6():
    return render_template('cgame6.html')

@app.route('/agame1')
def agame1():
    return render_template('agame1.html')

@app.route('/agame2')
def agame2():
    return render_template('agame2.html')

@app.route('/agame3')
def agame3():
    return render_template('agame3.html')

@app.route('/agame4')
def agame4():
    return render_template('agame4.html')

@app.route('/agame5')
def agame5():
    return render_template('agame5.html')

@app.route('/agame6')
def agame6():
    return render_template('agame6.html')

@app.route('/three')
def three():
    return render_template('three.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect('/three')
    return render_template('login.html')

@app.route('/toddler')
def toddler ():
    return render_template('toddler qn.html')
# Load models trained on 'tod_balanced.csv'
model_files = {
    "random_forest": "tod_random_forest.pkl",
    "decision_tree": "tod_decision_tree.pkl",
    "xgboost": "tod_xgboost.pkl"
}

models = {}
for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        try:
            models[model_name] = joblib.load(file_path)
            print(f"✅ {model_name} model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
    else:
        print(f"⚠️ Warning: {file_path} not found. {model_name} model not loaded.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure all required inputs are received, matching dataset features
        input_data = [int(request.form[f"A{i}"]) for i in range(1, 11)] + [
            int(request.form["Age_Mons"]),
            int(request.form["Qchat-10-Score"]),
            int(request.form["Sex"]),
            int(request.form["Ethnicity"]),
            int(request.form["Jaundice"]),
            int(request.form["Family_mem_with_ASD"])  # Fixed field name
        ]

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Get predictions from all models
        predictions = {name: model.predict(input_array)[0] for name, model in models.items()}

        # Determine ASD result based on majority vote
        asd_result = "✅ ASD Detected" if sum(predictions.values()) > 1 else "No ASD"

        # Get best model based on highest confidence
        best_model = max(predictions, key=predictions.get)

        return render_template("tod_result.html", 
                               asd_result=asd_result, 
                               best_model=best_model, 
                               predictions=predictions,
                               qchat_score=request.form["Qchat-10-Score"])
    except KeyError as e:
        missing_field = str(e).strip("'")
        return jsonify({"error": f"Missing required field: {missing_field}"}), 400
    except ValueError:
        return jsonify({"error": "Invalid input. Please enter numeric values."}), 400
    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/adult')
def adult():
    return render_template('adult qn.html')
try:
    dt_model = joblib.load("decision_tree.pkl")
    rf_model = joblib.load("random_forest.pkl")
    xgb_model = joblib.load("xgboost.pkl")
    scaler = joblib.load("scaler.pkl")
    model_features = joblib.load("model_features.pkl")
    print("✅ Models and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    dt_model = rf_model = xgb_model = scaler = model_features = None

@app.route('/predicta', methods=['POST'])
def predicta():
    try:
        # Ensure models and feature names are loaded
        if None in (dt_model, rf_model, xgb_model, scaler, model_features):
            return jsonify({"error": "Model, scaler, or feature names not loaded properly."})

        # Debug: Print received form data
        print("📩 Received form data:", request.form)

        # Extract and validate input features
        input_features = []
        for key in model_features:  # Ensure correct order
            try:
                input_features.append(float(request.form[key]))
            except ValueError:
                return jsonify({"error": f"Invalid input for {key}"})

        # Convert input to a DataFrame (to match training format)
        input_df = pd.DataFrame([input_features], columns=model_features)
        print("📊 Input DataFrame:")
        print(input_df.head())

        # Apply scaling
        input_scaled = scaler.transform(input_df)
        print("📊 Scaled Input:", input_scaled)

        # Get predictions from models
        dt_pred = dt_model.predict(input_scaled)[0]
        rf_pred = rf_model.predict(input_scaled)[0]
        xgb_pred = xgb_model.predict(input_scaled)[0]

        # Debug: Print model predictions
        print(f"📌 DT Prediction: {dt_pred}, RF Prediction: {rf_pred}, XGB Prediction: {xgb_pred}")

        # Store predictions in a list
        predictions = [dt_pred, rf_pred, xgb_pred]

        # Determine final prediction based on majority vote
        final_prediction = 1 if predictions.count(1) > predictions.count(0) else 0
        result_text = "✅ ASD Detected" if final_prediction == 1 else "❌ No ASD"

        print(f"🎯 Final Prediction: {result_text}")

# Determine the best model based on the highest confidence score
        best_model = max(zip(['Decision Tree', 'Random Forest', 'XGBoost'], predictions), key=lambda x: x[1])[0]

        return render_template('adult_result.html', result=result_text, predictions=predictions, best_model=best_model, input_values=input_features)

    except Exception as e:
        print(f"⚠️ Error during prediction: {str(e)}")
        return jsonify({"error": str(e)})
    
@app.route('/child')
def child ():
    return render_template('child qn.html')
# Define model paths
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

# Load the trained models using joblib
try:
    decision_tree_model = joblib.load(os.path.join(MODEL_PATH, "decision_tree_autism_children.pkl"))
    random_forest_model = joblib.load(os.path.join(MODEL_PATH, "random_forest_autism_children.pkl"))
    xgboost_model = joblib.load(os.path.join(MODEL_PATH, "xgboost_autism_children.pkl"))
except Exception as e:
    print(f"Error loading models: {e}")

@app.route("/predictc", methods=["POST"])
def predictc():
    try:
        # Debug print to check received form data
        print("Received form data:", request.form)
        
        # Expected features based on dataset
        expected_features = [
            "age", "gender", "ethnicity", "jundice", "relation", "used_app_before", 
            "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", 
            "A7_Score", "A8_Score", "A9_Score", "A10_Score"
        ]
        
        # Collect features safely
        features = [float(request.form[key]) if key in request.form and request.form[key] else 0 for key in expected_features]
        input_data = np.array([features])
        
        # Initialize predictions
        predictions = {}
        
        for model_name, model in zip(["Decision Tree", "Random Forest", "XGBoost"], 
                                     [decision_tree_model, random_forest_model, xgboost_model]):
            predictions[model_name] = int(model.predict(input_data)[0])
        
        # Determine the final prediction (majority vote)
        final_prediction = "Autistic" if sum(predictions.values()) > 1 else "Not Autistic"
        
        # Determine the best model based on majority vote
        best_model = max(predictions, key=predictions.get) if predictions else "Unknown"
        
        # Create result dictionary
        result = {
            "final_prediction": final_prediction,
            "best_model": best_model,
            "message": "Predictions generated successfully!",
            "results": predictions
        }
        
        # Render the result page with structured data
        return render_template("child_result.html", result=result)
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return render_template("child_result.html", result={"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
