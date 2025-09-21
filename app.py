from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import warnings

# Suppress sklearn warnings about version mismatch
warnings.filterwarnings('ignore', category=UserWarning)

# importing model with error handling
try:
    model = pickle.load(open('model.pkl','rb'))
    sc = pickle.load(open('standscaler.pkl','rb'))
    ms = pickle.load(open('minmaxscaler.pkl','rb'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure model files are compatible with your scikit-learn version.")

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Detailed crop information with reasoning
    crop_info = {
        "Rice": {
            "type": "Cereal Grain",
            "season": "Kharif (Monsoon)",
            "water_req": "High water requirement",
            "soil_type": "Clay/Loamy soil",
            "benefits": "Staple food crop, high carbohydrate content"
        },
        "Maize": {
            "type": "Cereal Grain", 
            "season": "Kharif/Rabi",
            "water_req": "Moderate water requirement",
            "soil_type": "Well-drained fertile soil",
            "benefits": "Versatile crop for food, feed, and industrial use"
        },
        "Cotton": {
            "type": "Cash Crop",
            "season": "Kharif",
            "water_req": "Moderate to high water requirement",
            "soil_type": "Black cotton soil",
            "benefits": "High economic value, textile industry"
        },
        "Coffee": {
            "type": "Plantation Crop",
            "season": "Perennial",
            "water_req": "Moderate water requirement",
            "soil_type": "Well-drained acidic soil",
            "benefits": "High export value, long-term investment"
        },
        "Coconut": {
            "type": "Plantation Crop",
            "season": "Perennial",
            "water_req": "Moderate water requirement",
            "soil_type": "Sandy loam soil",
            "benefits": "Multiple products: oil, fiber, water"
        },
        "Lentil": {
            "type": "Pulse Crop",
            "season": "Rabi",
            "water_req": "Low water requirement",
            "soil_type": "Well-drained soil",
            "benefits": "High protein content, nitrogen fixation"
        },
        "Chickpea": {
            "type": "Pulse Crop",
            "season": "Rabi",
            "water_req": "Low to moderate water requirement",
            "soil_type": "Well-drained soil",
            "benefits": "High protein, improves soil fertility"
        },
        "Apple": {
            "type": "Fruit Crop",
            "season": "Perennial",
            "water_req": "Moderate water requirement",
            "soil_type": "Well-drained loamy soil",
            "benefits": "High nutritional value, good market price"
        },
        "Mango": {
            "type": "Fruit Crop",
            "season": "Perennial",
            "water_req": "Moderate water requirement",
            "soil_type": "Deep well-drained soil",
            "benefits": "King of fruits, high market demand"
        },
        "Banana": {
            "type": "Fruit Crop",
            "season": "Perennial",
            "water_req": "High water requirement",
            "soil_type": "Rich loamy soil",
            "benefits": "Year-round production, high nutrition"
        },
        "Grapes": {
            "type": "Fruit Crop",
            "season": "Perennial",
            "water_req": "Moderate water requirement",
            "soil_type": "Well-drained soil",
            "benefits": "Wine production, fresh fruit market"
        },
        "Orange": {
            "type": "Fruit Crop",
            "season": "Perennial",
            "water_req": "Moderate water requirement",
            "soil_type": "Well-drained acidic soil",
            "benefits": "High vitamin C, processing industry"
        },
        "Papaya": {
            "type": "Fruit Crop",
            "season": "Perennial",
            "water_req": "Moderate water requirement",
            "soil_type": "Well-drained fertile soil",
            "benefits": "Fast growing, medicinal properties"
        },
        "Pomegranate": {
            "type": "Fruit Crop",
            "season": "Perennial",
            "water_req": "Low to moderate water requirement",
            "soil_type": "Well-drained soil",
            "benefits": "Antioxidant rich, drought tolerant"
        },
        "Watermelon": {
            "type": "Vegetable/Fruit",
            "season": "Summer",
            "water_req": "High water requirement",
            "soil_type": "Sandy loam soil",
            "benefits": "High water content, summer market"
        },
        "Muskmelon": {
            "type": "Vegetable/Fruit",
            "season": "Summer",
            "water_req": "Moderate water requirement",
            "soil_type": "Well-drained sandy soil",
            "benefits": "Good summer crop, high nutrition"
        }
    }

    def generate_reasoning(crop, N, P, K, temp, humidity, ph, rainfall):
        reasoning = f"Based on your soil and climate conditions:\n\n"
        
        # Nutrient analysis
        if N > 50:
            reasoning += f"✓ High nitrogen levels ({N:.1f}) support vigorous plant growth\n"
        elif N < 20:
            reasoning += f"⚠ Lower nitrogen levels ({N:.1f}) are suitable for legumes that fix their own nitrogen\n"
        else:
            reasoning += f"✓ Moderate nitrogen levels ({N:.1f}) are ideal for most crops\n"
            
        if P > 30:
            reasoning += f"✓ Good phosphorus content ({P:.1f}) promotes root development\n"
        else:
            reasoning += f"✓ Phosphorus levels ({P:.1f}) are adequate for this crop\n"
            
        if K > 40:
            reasoning += f"✓ High potassium ({K:.1f}) enhances disease resistance and fruit quality\n"
        else:
            reasoning += f"✓ Potassium levels ({K:.1f}) support healthy plant metabolism\n"
        
        # Climate analysis
        if temp > 30:
            reasoning += f"🌡️ High temperature ({temp:.1f}°C) suits warm-season crops\n"
        elif temp < 15:
            reasoning += f"🌡️ Cool temperature ({temp:.1f}°C) is ideal for cool-season crops\n"
        else:
            reasoning += f"🌡️ Moderate temperature ({temp:.1f}°C) provides optimal growing conditions\n"
            
        if humidity > 70:
            reasoning += f"💧 High humidity ({humidity:.1f}%) creates favorable moisture conditions\n"
        elif humidity < 40:
            reasoning += f"💧 Low humidity ({humidity:.1f}%) suits drought-tolerant varieties\n"
        else:
            reasoning += f"💧 Moderate humidity ({humidity:.1f}%) is ideal for most crops\n"
            
        if ph > 7.5:
            reasoning += f"🧪 Alkaline soil (pH {ph:.1f}) is well-suited for this crop\n"
        elif ph < 6.5:
            reasoning += f"🧪 Acidic soil (pH {ph:.1f}) matches this crop's preferences\n"
        else:
            reasoning += f"🧪 Neutral soil pH ({ph:.1f}) provides optimal nutrient availability\n"
            
        if rainfall > 1000:
            reasoning += f"🌧️ High rainfall ({rainfall:.1f}mm) supports water-intensive crops\n"
        elif rainfall < 500:
            reasoning += f"🌧️ Lower rainfall ({rainfall:.1f}mm) suits drought-resistant varieties\n"
        else:
            reasoning += f"🌧️ Moderate rainfall ({rainfall:.1f}mm) provides adequate water supply\n"
            
        return reasoning

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        reasoning = generate_reasoning(crop, N, P, K, temp, humidity, ph, rainfall)
        
        crop_details = crop_info.get(crop, {
            "type": "Agricultural Crop",
            "season": "Seasonal",
            "water_req": "Moderate",
            "soil_type": "Various soil types",
            "benefits": "Good agricultural choice"
        })
        
        result = {
            "crop": crop,
            "reasoning": reasoning,
            "details": crop_details
        }
    else:
        result = {
            "crop": "Unknown",
            "reasoning": "Sorry, we could not determine the best crop with the provided data.",
            "details": {}
        }
    
    return render_template('index.html', result=result)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

# For Vercel deployment
def handler(request):
    return app(request.environ, lambda *args: None)

# python main
if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)