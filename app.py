from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lower
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%pip install openpyxl

# Initialize Spark session
spark = SparkSession.builder.appName("DataFrameExample").getOrCreate()

file_path = "/dbfs/FileStore/deliminated.xlsx"
co_occurance_df = pd.read_excel(file_path)

# Fill NaN values in the Pandas DataFrame
co_occurance_df = co_occurance_df.fillna("")

# Convert the Pandas DataFrame to a Spark DataFrame
co_occurance_spark_df = spark.createDataFrame(co_occurance_df)

# Fill NaN values in the Spark DataFrame
co_occurance_spark_df = co_occurance_spark_df.fillna("")

# Display the Spark DataFrame
#display(co_occurance_spark_df.count())

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Lowercase column names
co_occurance_spark_df = co_occurance_spark_df.select([lower(col(c)).alias(c) for c in co_occurance_df.columns])

# Create an RDD where each row is a set of words
rdd = co_occurance_spark_df.rdd.map(lambda row: set(filter(None, row)))

# Generate word pair co-occurrences
co_occurrences = rdd.flatMap(lambda words: [(tuple(sorted(pair)), 1) for pair in combinations(words, 2)])

# Aggregate counts
co_occurrence_df = spark.createDataFrame(co_occurrences, ["word_pair", "count"]) \
    .groupBy("word_pair").sum("count") \
    .withColumnRenamed("sum(count)", "frequency") \
    .orderBy(col("frequency").desc())

# Show the top 20 word pairs
co_occurrence_df.show(100, truncate=False)
display(co_occurrence_df.limit(100))

# Save results as a new table in Databricks
co_occurrence_df.write.mode("overwrite").saveAsTable("word_co_occurrence_analysis")

# Assuming co_occurrence_df.limit(100) has been displayed and we are starting from here
# Convert the Spark DataFrame to a Pandas DataFrame for visualization
top_20_pd_df = co_occurrence_df.limit(20).toPandas()

# Split word pairs into separate columns
top_20_pd_df[['word1', 'word2']] = pd.DataFrame(top_20_pd_df['word_pair'].tolist(), index=top_20_pd_df.index)

# Create a pivot table for the heatmap
pivot_table = top_20_pd_df.pivot_table(index='word1', columns='word2', values='frequency', fill_value=0)

# Filter out rows and columns with all zeros
pivot_table = pivot_table.loc[(pivot_table != 0).any(axis=1), (pivot_table != 0).any(axis=0)]

# Generate the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_table, annot=True, fmt="g", cmap="viridis")
plt.title("Top 20 Word Pair Co-occurrence Heatmap")
plt.show()
