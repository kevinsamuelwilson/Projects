'''
Usage:
    $ spark-submit hypyerparameterTuning.py train_file val_file

    e.g.
    $ spark-submit hypyerparameterTuning.py reviews_poetry_train.parquet reviews_poetry_val.parquet
    or
    $ spark-submit hypyerparameterTuning.py reviews_small_train.parquet reviews_small_val.parquet
'''

import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator

def hyperparameter_tuning(spark, train_file, val_file):

    # load in the data
    train = spark.read.parquet(train_file)
    train.createOrReplaceTempView('train')
    val = spark.read.parquet(val_file)
    val.createOrReplaceTempView('val')
    user_idxer = StringIndexer(inputCol = 'user_id', outputCol = 'user', handleInvalid = "skip")
    item_idxer = StringIndexer(inputCol = 'book_id', outputCol = 'item', handleInvalid = "skip")
    pipeline = Pipeline(stages = [user_idxer, item_idxer])
    indexers = pipeline.fit(train)
    train = indexers.transform(train)
    val = indexers.transform(val)
    val = val.withColumn('item', val['item'].cast('int'))
    val = val.withColumn('user', val['user'].cast('int'))
    val_users = val.select('user').distinct()
    val_groundtruth = val.groupby('user').agg(F.collect_list('item').alias('truth')).cache()

    # ranks to test
    # ranks = [1,2,5,10,50]
    ranks = [10]

    # regParams to test
    # lambdas = [0.01, 0.1, 1, 2, 10]
    lambdas = [0.01]

    # Set up list for results
    p = []
    iters = len(ranks) * len(lambdas)
    count = 0

    for r in ranks:
        for lam in lambdas:
            print('regParam: {}, Rank: {}'.format(lam, r))
            als = ALS(regParam = lam, rank = r,
                userCol='user', itemCol='item', seed=2020, ratingCol='rating',
                nonnegative=True, coldStartStrategy='drop',
                intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK')
            model = als.fit(train)
            rec = model.recommendForAllUsers(500)
            predictions = rec.join(val_groundtruth, rec.user == val_groundtruth.user, 'inner')
            predictions = predictions.select('recommendations.item', 'truth')
            predictionAndLabels = predictions.rdd.map(tuple).repartition(1000)
            metrics = RankingMetrics(predictionAndLabels)
            precision = metrics.precisionAt(500)
            MAP = metrics.meanAveragePrecision

            p.append([lam, r, MAP, precision, model, als])
            count += 1
            print('precision: {}, MAP: {}'.format(precision, MAP))
            print('done with iter {} out of {}'.format(count, iters))



# Enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('spark-goodreads-recsys').getOrCreate()

    # Get train_file and val_file from command line

    train_file = sys.argv[1]
    # train_file = 'reviews_poetry_train.parquet'

    val_file = sys.argv[2]
    # val_file = 'reviews_poetry_val.parquet'

    # Call model eval routine
    hyperparameter_tuning(spark, train_file, val_file)
