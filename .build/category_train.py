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

print("Spark NLP version, expecting 4.3.1:  ", sparknlp.version())
print("Apache Spark version, expecting 3.2.3:  ", spark.version)

# Load dataset
train = spark.read.json("dataset_en_train.json")

# Samples per class for rebalancing
class_samples = 5000

# Resample each class to the size of the class samples with replacement
train_resampled_s = []
for c in train.select("product_category").distinct().collect():
    classDF = train.filter(train["product_category"] == c[0])
    train_resampled = classDF.sample(True, class_samples / classDF.count(), seed=42)
    train_resampled_s.append(train_resampled)

# Concatenate the resampled DataFrames
train_balanced = train_resampled_s[0]
for df in train_resampled_s[1:]:
    train_balanced = train_balanced.union(df)

# Build pipeline
document_assembler = DocumentAssembler() \
      .setInputCol("review_body") \
      .setOutputCol("document")
    
tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")
      
normalizer = Normalizer() \
      .setInputCols(["token"]) \
      .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

stemmer = Stemmer() \
      .setInputCols(["cleanTokens"]) \
      .setOutputCol("stem")

finisher = Finisher() \
      .setInputCols(["stem"]) \
      .setOutputCols(["token_features"]) \
      .setOutputAsArray(True) \
      .setCleanAnnotations(False)

countVectors = CountVectorizer(inputCol="token_features", outputCol="features", vocabSize=10000, minDF=5)
label_stringIdx = StringIndexer(inputCol = "product_category", outputCol = "label")

nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            countVectors,
            label_stringIdx])

# Fit pipeline
nlp_model = nlp_pipeline.fit(train_balanced)
train_processed = nlp_model.transform(train_balanced)

# Save fitted pipeline
nlp_model.write().overwrite().save("fitted_pipeline_category")

# Classifier
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(train_processed)

# Save model
lrModel.write().overwrite().save("lrmodel_category")