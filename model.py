from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Create SparkSession
spark = SparkSession.builder \
    .appName("MusicRecommendationModel") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/fma.audio_features") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/fma.audio_features") \
    .getOrCreate()

# Load data from MongoDB
df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

# Load historical data from MongoDB
history_df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load("history")
print("Schema of the historical data:")
history_df.printSchema()
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

# Join historical data with the main DataFrame
df = df.join(history_df, df.filename == history_df.file_path, "left_outer")

# Replace NaN values with zeros
df = df.fillna(0)

# Split data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train ALS model
als = ALS(maxIter=10, regParam=0.01, userCol="filename_id", itemCol="filename_id", ratingCol="spectral_centroid_mean",
          coldStartStrategy="drop")

model = als.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="spectral_centroid_mean", predictionCol="prediction")

try:
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE):", rmse)
except Exception as e:
    print("An error occurred while evaluating the model:", str(e))

# Save the model
model.save("music_recommendation_model")
