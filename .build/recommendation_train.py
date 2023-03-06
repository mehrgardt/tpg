# Imports
import sparknlp

from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import concat,col

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row

# Start sparknlp
spark = sparknlp.start()

print("Spark NLP version, expecting 4.3.1:  ", sparknlp.version())
print("Apache Spark version, expecting 3.2.3:  ", spark.version)

# Load dataset
train = spark.read.json("dataset_en_train.json")

# Fit pipeline
stringIndexer = StringIndexer(inputCol="product_id", outputCol="product_id_int")
nlp_model = stringIndexer.fit(train)
train_processed = nlp_model.transform(train)

stringIndexer = StringIndexer(inputCol="reviewer_id", outputCol="reviewer_id_int")
nlp_model = stringIndexer.fit(train_processed)
train_processed = nlp_model.transform(train_processed)

train_processed = train_processed.withColumn("stars", col("stars").cast("int"))

als = ALS(maxIter=5, regParam=0.01, userCol="reviewer_id_int", itemCol="product_id_int", ratingCol="stars",
          coldStartStrategy="drop")
model = als.fit(train_processed)

# Save model
model.write().overwrite().save("alsmodel_recommendation")