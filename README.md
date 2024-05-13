# Music Streaming

This project aims to develop a streamlined alternative to Spotify, featuring a music recommendation system, playback, and streaming capabilities, alongside real-time suggestions derived from user activity. The project is divided into several phases, each with its specific tasks and objectives.

Phase 1: Extract, Transform, Load (ETL) Pipeline
The first phase involves creating an Extract, Transform, Load (ETL) pipeline utilizing the Free Music Archive (FMA) dataset. This dataset comprises over 100,000 tracks spanning various genres. The tasks in this phase include:

Extracting audio features from the FMA dataset, such as Mel-Frequency Cepstral Coefficients (MFCC), spectral centroid, and zero-crossing rate.
Transforming the audio data into numerical and vector formats using Python libraries like librosa.
Storing the transformed audio features in a scalable and accessible manner using MongoDB.
Files:
preprocessing.py: Python script for extracting audio features and storing them in MongoDB.

Phase 2: Music Recommendation Model
In this phase, Apache Spark is utilized to train a music recommendation model. The model now uses the K-Means clustering algorithm for enhanced accuracy. The tasks include:

1) Training a recommendation model using Apache Spark's MLlib with the K-Means clustering algorithm.
2) Evaluating the model's performance using relevant metrics.
3) Saving the trained model for future use.

Files:
model.py: Python script for training a music recommendation model using Apache Spark.
Phase 3: Deployment
The deployment phase involves building an interactive music streaming web application. The application dynamically generates music recommendations for users in real-time using Apache Kafka. The tasks include:

1) Developed a user-friendly web interface using frameworks like Flask.
2) Implementing real-time recommendation generation using Apache Kafka and historical playback data.
3) Ensuring seamless streaming and playback capabilities within the web application.
Files:
producer.py: Python script for generating real-time music recommendations using Apache Kafka.
app.py: Flask application for streaming music and displaying recommendations.


Conclusion
This project aimed to create a personalized music streaming experience akin to Spotify, incorporating a music recommendation system, playback, and streaming capabilities.

Through a series of phases, the project successfully accomplished the following key objectives:

Phase 1: Established an Extract, Transform, Load (ETL) pipeline to extract audio features from the Free Music Archive (FMA) dataset, transform them into numerical formats, and store them in a scalable MongoDB database.

Phase 2: Utilized Apache Spark to train a music recommendation model, employing K-Means algorithm for accurate recommendations. The model's performance was evaluated using Silhouette Score, and the trained model was saved for future use.

Phase 3: Developed an interactive music streaming web application with a user-friendly interface using Flask. Real-time music recommendations were generated using Apache Kafka, leveraging historical playback data to tailor suggestions dynamically.

By seamlessly integrating data processing, machine learning, real-time streaming, and web development technologies, this project showcases a holistic approach to building a personalized music streaming platform. Through continuous learning and adaptation to user feedback, the platform aims to provide a rich and engaging music discovery experience tailored to each user's preferences.

In summary, this project demonstrates the potential of leveraging advanced technologies to create innovative solutions in the realm of digital music streaming, offering users a unique and personalized music listening experience.
