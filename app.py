from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import time

app = Flask(__name__)

# Load trained model
model = load_model("mushroom_model.h5")

# Class labels (must match training order)
class_names = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius',
               'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

# Genus-based poison probabilities (heuristic knowledge)
genus_poison_prob = {
    "Amanita": 0.9,
    "Cortinarius": 0.8,
    "Entoloma": 0.7,
    "Russula": 0.4,
    "Boletus": 0.3,
    "Lactarius": 0.3,
    "Suillus": 0.2,
    "Agaricus": 0.2,
    "Hygrocybe": 0.2
}

# Prediction function
def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    # Top 3 predictions
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(class_names[i], preds[i]) for i in top3_idx]

    # Poisonous score calculation
    poisonous = 0
    for i in range(len(preds)):
        genus = class_names[i]
        poisonous += preds[i] * genus_poison_prob.get(genus, 0.5)

    edible = 1 - poisonous
    top_genus = class_names[np.argmax(preds)]

    return top_genus, edible, poisonous, top3


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = "static/upload.jpg"
        file.save(file_path)

        genus, edible, poisonous, top3 = predict(file_path)

        if poisonous <= 0.3:
            decision = "Likely Edible (Low Risk)"
            color = "green"
        elif poisonous >= 0.7:
            decision = "Likely Poisonous (High Risk)"
            color = "red"
        else:
            decision = "Uncertain — Do Not Consume"
            color = "orange"

        return render_template(
            "index.html",
            result=True,
            genus=genus,
            edible=round(edible*100, 2),
            poisonous=round(poisonous*100, 2),
            top3=top3,
            decision=decision,
            color=color,
            image_path=f"/{file_path}?v={int(time.time())}"
        )

    return render_template("index.html", result=False)

if __name__ == "__main__":
    app.run(debug=True)