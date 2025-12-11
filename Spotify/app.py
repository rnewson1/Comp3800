import numpy as np
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)

# Load the trained model once when the app starts
with open("gnb_model.pkl", "rb") as f:
    model = pickle.load(f)

@flask_app.route("/", methods=["GET", "POST"])
def index():
    predicted_text = ""

    if request.method == "POST":
        try:
            # 1. Read and convert inputs from the form
            streams = float(request.form["streams"])
            in_spotify_playlists = float(request.form["in_spotify_playlists"])
            in_apple_playlists = float(request.form["in_apple_playlists"])
            in_deezer_playlists = float(request.form["in_deezer_playlists"])
            bpm = float(request.form["bpm"])
            danceability = float(request.form["danceability_%"])
            valence = float(request.form["valence_%"])
            energy = float(request.form["energy_%"])
            acousticness = float(request.form["acousticness_%"])
            instrumentalness = float(request.form["instrumentalness_%"])
            liveness = float(request.form["liveness_%"])
            speechiness = float(request.form["speechiness_%"])

            # 2. Build feature vector in the SAME order you used in training
            # Example order; change to match your training code:
            features = np.array([[
                streams,
                in_spotify_playlists,
                in_apple_playlists,
                in_deezer_playlists,
                bpm,
                danceability,
                valence,
                energy,
                acousticness,
                instrumentalness,
                liveness,
                speechiness
            ]])

            # 3. Predict with your model
            prediction = model.predict(features)[0]

            # 4. Turn numeric prediction into a readable label
            # Adjust this mapping to match how your target variable is encoded
            if prediction == 1:
                predicted_text = "Popular"
            else:
                predicted_text = "Not popular"

        except Exception as e:
            # Basic error reporting; useful while you debug
            predicted_text = f"Error while predicting: {e}"

    # 5. Render template with prediction (empty on first GET)
    return render_template("index.html", predicted_text=predicted_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
