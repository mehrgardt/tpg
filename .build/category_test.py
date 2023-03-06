# Imports
import sparknlp

from sparknlp.base import *
#from sparknlp.common import *
#from sparknlp.annotator import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, OneHotEncoder, StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#from pyspark.sql.functions import concat,col

# Start sparknlp
spark = sparknlp.start()

# Load dataset
test = spark.read.json("dataset_en_test.json")

# Load PipelineModel
nlp_model = PipelineModel.load("fitted_pipeline_category")

# Load LogisticRegressionModel
lrModel = LogisticRegressionModel.load("lrmodel_category")

# Transform test data
test_processed = nlp_model.transform(test)

# predict test data
predictions = lrModel.transform(test_processed)

# Show test data
predictions.filter(predictions['prediction'] == 0) \
    .select("product_category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

# Evaluate test data
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(predictions))