import os
import sys
import platform
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, trim, avg, count, round

if platform.system() == "Windows":
    os.environ['HADOOP_HOME'] = "C:\\hadoop"

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.common import read_config

class AmazonDataPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)

        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        
        self.raw_data_dir = os.path.join(self.base_dir, self.config["paths"]["raw_data_dir"])
        self.processed_data_path = os.path.join(self.base_dir, self.config["paths"]["processed_data"])
        
        self.meta_file = self.config["files"]["meta_data"]
        self.reviews_file = self.config["files"]["reviews_data"]

        app_name = self.config["spark"]["app_name"] if "spark" in self.config else "Amazon_Fashion_ETL"
        memory = self.config["spark"]["driver_memory"] if "spark" in self.config else "4g"

        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", memory) \
            .config("spark.sql.caseSensitive", "true") \
            .getOrCreate()
        
        print(f"PySpark Engine Started: {app_name} (with permission from {memory} RAM)")
    
    def process_data(self):
        
        meta_path = os.path.join(self.raw_data_dir, self.meta_file) 
        reviews_path = os.path.join(self.raw_data_dir, self.reviews_file)
        
        print(f"Reading Meta Data from : {meta_path}")
        print(f"Reading Review Data from : {reviews_path}")

        df_meta = self.spark.read.option("mode", "DROPMALFORMED").json(meta_path)
        df_reviews = self.spark.read.option("mode", "DROPMALFORMED").json(reviews_path)

        df_meta_clean = df_meta.filter((col("title").isNotNull()) & (col("title") != ""))
        df_meta_clean = df_meta_clean.withColumn(
            "document",
            trim(concat_ws(" - ", col("title"), concat_ws(" ", col("description") ) ))
        )
        
        df_meta_clean = df_meta_clean.drop("average_rating", "rating_number", "review_count")
        
        print("Aggregating reviews...")
        
        df_reviews_agg = df_reviews.groupBy("parent_asin").agg(
            round(avg("rating"), 1).alias("average_rating"),
            count("rating").alias("review_count")
        )

        print("Joining Meta and Review datasets...")
        df_joined = df_meta_clean.join(df_reviews_agg, on="parent_asin", how="left")
        df_joined = df_joined.fillna({"average_rating": 0.0, "review_count": 0})
        
        df_final = df_joined.select("parent_asin", "title", "document", "price", "average_rating", "review_count")
        
        total_rows = df_final.count()
        print(f"Cleaning & Joining Has Done ! \nTotal Number Of Products To Be Processed: {total_rows}")
        
        df_final.write.mode("overwrite").parquet(self.processed_data_path)
        print(f"The enriched data is saved in Parquet format: {self.processed_data_path}")
        
        return df_final

if __name__ == "__main__":
    pipeline = AmazonDataPipeline()
    pipeline.process_data()