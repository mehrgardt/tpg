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

# Load dataset
test = spark.read.json("dataset_en_test.json")

stringIndexer = StringIndexer(inputCol="product_id", outputCol="product_id_int")
nlp_model = stringIndexer.fit(test)
test_processed = nlp_model.transform(test)

stringIndexer = StringIndexer(inputCol="reviewer_id", outputCol="reviewer_id_int")
nlp_model = stringIndexer.fit(test_processed)
test_processed = nlp_model.transform(test_processed)

test_processed = test_processed.withColumn("stars", col("stars").cast("int"))

#test_processed.groupBy('product_id_int').count().orderBy('count', ascending=False).show(10)

# Load ALSModel
model = ALSModel.load("alsmodel_recommendation")

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test_processed)

# Recommend 5 products for all reviewers
userRecs = model.recommendForAllUsers(5)
#userRecs.show()