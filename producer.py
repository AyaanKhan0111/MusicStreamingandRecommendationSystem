from kafka import KafkaProducer
import json
import time
from pymongo import MongoClient
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                         acks='all')

# Check if the producer is connected to the broker
if not producer.bootstrap_connected():
    print("Producer failed to connect to Kafka broker.")
    exit(1)

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['fma']
collection = db['audio_features']

# Load the ALS model
spark = SparkSession.builder \
    .appName("MusicRecommendationModel") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/fma.audio_features") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/fma.audio_features") \
    .getOrCreate()

model = ALSModel.load("music_recommendation_model")

# Simulate user activity and produce recommendations
while True:
    try:
        # Retrieve historical data from MongoDB
        history = collection.find().limit(100)  # Assuming you want to consider the last 100 interactions
        
        # Convert historical data to DataFrame
        history_df = spark.createDataFrame(history)

        # Generate recommendations using the ALS model
        recommendations = model.recommendForUserSubset(history_df, 5)  # Generate 5 recommendations per user
        
        # Convert recommendations to JSON and produce to Kafka topic
        for row in recommendations.collect():
            user_id = row.filename_id
            recommended_songs = [r.filename for r in row.recommendations]
            producer.send('music_recommendations', value={"user_id": user_id, "recommended_songs": recommended_songs})
            print("Recommendations sent for user", user_id, ":", recommended_songs)
            
        time.sleep(1)  # Simulate a short delay between sending recommendations
    except Exception as e:
        print("Error:", e)
        # You can handle the error here, e.g., retrying or logging
