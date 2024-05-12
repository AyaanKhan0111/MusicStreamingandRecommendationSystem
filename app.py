from flask import Flask, render_template, Response, jsonify
from kafka import KafkaConsumer
import json
import os

app = Flask(__name__)

# Define the base path for the audio files directory
audio_files_base_path = '/media/hdoop/TOSHIBA EXT/fma_large_sample/'

# Initialize Kafka consumer
consumer = KafkaConsumer('music_recommendations',
                         group_id='music-recommendation-group',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

# Function to list audio files from the specified directories
def list_audio_files():
    audio_files = []
    for folder_name in os.listdir(audio_files_base_path):
        folder_path = os.path.join(audio_files_base_path, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.mp3'):
                    audio_files.append(os.path.join(folder_name, filename))
    return audio_files

# Function to fetch recommendations from Kafka
def fetch_recommendations():
    recommendations = []
    for message in consumer:
        # Extract file name from the recommendation
        file_name = message.value['filename']
        recommendations.append(file_name)
        if len(recommendations) >= 5:  # Limiting to 5 recommendations
            break
    return recommendations

# Route for home page
@app.route('/')
def index():
    # Get the list of audio files
    audio_files = list_audio_files()

    # Fetch recommendations
    recommendations = fetch_recommendations()

    return render_template('index.html', audio_files=audio_files, recommendations=recommendations)

# Flask route for streaming music
@app.route('/stream_music/<path:file_path>')
def stream_music(file_path):
    # Full path to the requested audio file
    requested_audio_path = os.path.join(audio_files_base_path, file_path)

    # Check if the file exists
    if not os.path.isfile(requested_audio_path):
        return "Audio file not found", 404

    # Open and stream the audio file
    with open(requested_audio_path, 'rb') as audio_file:
        return Response(audio_file.read(), mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(debug=True)
