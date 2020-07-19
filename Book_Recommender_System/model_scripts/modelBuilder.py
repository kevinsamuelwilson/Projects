'''
Usage:

First, set the hyperparameters in the ranks list and lambdas list in this file.
Then in the pyspark terminal:

>>> import modelBuilder as mb
>>> # train a set of models
>>> mb.train_model(spark, train_file, val_file)

To evaluate a  model
>>> mb.evaluate_model(spark, model_file, train_file, val_file)

This will print out your RMSE, Precision@500, MAP, and NDCG
'''

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import explode
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import collect_list
import itertools

def train_model (spark, train_file, val_file):
    '''
    Trains an ALS model on the given data and saves it to a model file
    '''
    
    # load in the data
    train = spark.read.parquet(train_file)
    train.createOrReplaceTempView('train')
    
    val = spark.read.parquet(val_file)
    val.createOrReplaceTempView('val')
    
    # use stringIndexer and pipeline to index on user_id and book_id
    user_idxer = StringIndexer(inputCol = 'user_id', outputCol = 'user', handleInvalid = 'skip')
    item_idxer = StringIndexer(inputCol = 'book_id', outputCol = 'item', handleInvalid = 'skip')
    
    
    # index data
    pipeline = Pipeline(stages = [user_idxer, item_idxer])
    indexer = pipeline.fit(train)
    train = indexer.transform(train)
    val = indexer.transform(val)
    
    # build parameter grid
    ranks = [100]
    lambdas = [0.001, 0.005, 0.008]
    
    # test different params
    for rank, lam in itertools.product(ranks, lambdas):
        als = ALS(rank = rank, regParam = lam, seed = 2020, numUserBlocks = 100, numItemBlocks = 100, coldStartStrategy = 'drop')
        model = als.fit(train)
        model.save('rank_' + str(rank) + '_lam_' + str(lam) + '_model.m')
        
    

def get_val_metrics(model, val):
    preds = model.transform(val)
    recs = model.recommendForUserSubset(val, 500)
    
    top_items = recs.selectExpr('user as user', 'recommendations.item as top_items')
    true_items = val.where(val.rating >= 3).groupby('user').agg(collect_list('item').alias('true_item_list'))
    predictions_and_labels_rankings = top_items.join(true_items, how = 'inner', on = 'user')\
        .select('true_item_list', 'top_items')
    
    predictions_and_labels_rankings.write.json('val_recs.json')
    
    ranking_metrics = RankingMetrics(predictions_and_labels_rankings.cache().rdd)
    prec_at = ranking_metrics.precisionAt(500)
    mean_avg_prec = ranking_metrics.meanAveragePrecision
    ndcg = ranking_metrics.ndcgAt(500)
    
    rmse = RegressionMetrics(preds.select('rating', 'prediction').cache().rdd).rootMeanSquaredError
    evaluator = RegressionEvaluator(predictionCol = 'prediction', labelCol = 'rating', metricName = 'rmse')
    rmse = evaluator.evaluate(preds)
    return rmse, prec_at, mean_avg_prec, ndcg
    

def evaluate_model(spark, model_file, train_file, val_file):
    # load in the data
    train = spark.read.parquet(train_file)
    train.createOrReplaceTempView('train')
    
    val = spark.read.parquet(val_file)
    val.createOrReplaceTempView('val')
    
    # use stringIndexer and pipeline to index on user_id and book_id
    user_idxer = StringIndexer(inputCol = 'user_id', outputCol = 'user', handleInvalid = 'skip')
    item_idxer = StringIndexer(inputCol = 'book_id', outputCol = 'item', handleInvalid = 'skip')
    
    
    # index data
    pipeline = Pipeline(stages = [user_idxer, item_idxer])
    indexer = pipeline.fit(train)
    train = indexer.transform(train)
    val = indexer.transform(val)
    
    # load the model
    model = ALSModel.load(model_file)
    rmse, prec_at, mean_avg_prec, ndcg  = get_val_metrics(model, val)
    
    # print('Rank = %d, lambda = %.2f' %(model.getRank(), model.getRegParam()))
    print('RMSE: %f, precision at 500: %f, MAP %f, ndcg at 500 %f' %(rmse, prec_at, mean_avg_prec, ndcg))



def get_val_metrics_outdated (spark, model_file, train_file, val_file, output_log_filepath):
    '''
    Gets val metrics for given model, training, and validation data and saves to a log file
    '''
    train = spark.read.parquet(train_file)
    train.createOrReplaceTempView('train')
    
    val = spark.read.parquet(val_file)
    val.createOrReplaceTempView('val')
    
    # use stringIndexer and pipeline to index on user_id and book_id
    # for user indexer, throws error if there is a user in the validation that is not in training
    # for item indexer, skips item if it was not in training 
    user_idxer = StringIndexer(inputCol = 'user_id', outputCol = 'user', handleInvalid = 'skip')
    item_idxer = StringIndexer(inputCol = 'book_id', outputCol = 'item', handleInvalid = 'skip')
    
    
    # index data
    pipeline = Pipeline(stages = [user_idxer, item_idxer])
    
    val = pipeline.fit(train).transform(val)
    
    alsmodel = ALSModel.load(model_file)
    
    preds = alsmodel.transform(val)
    recs = alsmodel.recommendForUserSubset(val, 400)
    top_items = recs.selectExpr('user as user', 'recommendations.item as top_items')
    
    good_preds = preds.where(preds.rating >= 3.0)
    recs_tall = recs.select(recs.user, explode(recs.recommendations))
    recs_taller = recs_tall.withColumn('item', recs_tall.col.item).withColumn('pred_rating', recs_tall.col.rating).select('user', 'item', 'pred_rating')
    
    joined = recs_taller.join(good_preds, how = 'inner', on = ['item', 'user'])
    
    return preds, recs, val