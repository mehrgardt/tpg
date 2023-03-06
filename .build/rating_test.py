# Imports
import sparknlp

from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, OneHotEncoder, StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import concat,col

# Start sparknlp
spark = sparknlp.start()

# Load dataset
test = spark.read.json("dataset_en_test.json")

# Load PipelineModel
nlp_model = PipelineModel.load("fitted_pipeline_rating")

# Load LogisticRegressionModel
lrModel = LogisticRegressionModel.load("lrmodel_rating")

# Concat review_title and review_body
test_rating = test.withColumn("review_body", concat(test.review_title, test.review_body))

# Transform test data
test_processed = nlp_model.transform(test_rating)

# predict test data
predictions = lrModel.transform(test_processed)

# Show test data
#predictions.filter(predictions['prediction'] == 0) \
#    .select("review_body","stars","probability","label","prediction") \
#    .orderBy("probability", ascending=False) \
#    .show(n = 10, truncate = 30)

# Evaluate test data
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(predictions))