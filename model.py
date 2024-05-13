from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Create SparkSession
spark = SparkSession.builder \
    .appName("MusicClusteringModel") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/fma.audio_features") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/fma.audio_features") \
    .getOrCreate()

# Load data from MongoDB
df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

# Check if the columns exist before exploding
if "mfccs_mean" in df.columns and "mfccs_std" in df.columns:
    # Explode the arrays and select relevant columns
    df = df.withColumn("mfccs_mean_val", explode(col("mfccs_mean"))) \
        .withColumn("mfccs_std_val", explode(col("mfccs_std"))) \
        .select("filename", "spectral_centroid_mean", "zero_crossing_rate_mean", "mfccs_mean_val", "mfccs_std_val")
else:
    print("Columns 'mfccs_mean' or 'mfccs_std' do not exist in the DataFrame. Selecting relevant columns that exist.")
    # Select relevant columns
    df = df.select("filename", "spectral_centroid_mean", "zero_crossing_rate_mean")

# Add a unique numeric identifier for each filename
df = df.withColumn("filename_id", monotonically_increasing_id())

# Define input columns for VectorAssembler
input_cols = ["spectral_centroid_mean", "zero_crossing_rate_mean"]

# Convert features to a vector
assembler = VectorAssembler(
    inputCols=input_cols,
    outputCol="features")

try:
    df = assembler.transform(df)
    print("DataFrame Schema after transformation:")
    df.printSchema()
except Exception as e:
    print("An error occurred while transforming the DataFrame:", str(e))
    # Exit the script if an error occurs
    exit()

# Split data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train KMeans model
kmeans = KMeans(k=10, seed=42)  # Adjust the number of clusters (k) as needed
model = kmeans.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = ClusteringEvaluator(metricName="silhouette", distanceMeasure="squaredEuclidean")

try:
    silhouette_score = evaluator.evaluate(predictions)
    print("Silhouette Score:", silhouette_score)
except Exception as e:
    print("An error occurred while evaluating the model:", str(e))

# Show cluster centers
print("Cluster Centers:")
centers = model.clusterCenters()
for center in centers:
    print(center)

# Save the model
model.save("music_clustering_model")
