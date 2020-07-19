module load python/gnu/3.6.5
module load spark/2.4.0
pyspark

########################################################################
#load full reviews
reviews = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv',
        header=True, schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
reviews.createOrReplaceTempView('reviews')
reviews = reviews.filter(reviews.is_read == 1)  #.filter(interactions.rating != 0.0)

#downsample
unique_users = spark.sql('SELECT user_id FROM reviews GROUP BY user_id HAVING COUNT(rating) >= 10')
unique_users.createOrReplaceTempView('unique_users')

unique_books = spark.sql('SELECT book_id FROM reviews GROUP BY book_id HAVING COUNT(rating) >= 10')
unique_books.createOrReplaceTempView('unique_books')

unique_users_small = unique_users.sample(False, 0.01, seed=10)
unique_books_small = unique_books.sample(False, 0.01, seed=10)

reviews_viz = reviews.join(unique_users_small, how = 'right', on = 'user_id').select('user_id', 'book_id', 'rating')
reviews_viz2 = reviews_viz.join(unique_books_small, how = 'inner', on = 'book_id').select('user_id', 'book_id', 'rating')
reviews_viz2.repartition('book_id').write.parquet('reviews_viz3.parquet')

reviews = spark.read.parquet('reviews_viz3.parquet')
reviews.createOrReplaceTempView('reviews')
bookUserDf = reviews.groupBy('book_id').pivot('user_id').sum('rating')
bookUserDf.write.parquet('bookUserDf.parquet')


########################################################################

spark.conf.set('spark.sql.pivotMaxValues', u'718022')
spark.conf.set("spark.executor.memory", "15G")
spark.conf.set("spark.driver.memory", "15G")
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


from pyspark.sql.functions import sum
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

reviews_file = 'reviews_viz2.parquet'
reviews_file = 'reviews_downsampled.parquet'
# reviews_file = 'reviews_smal_train.parquet'

reviews = spark.read.parquet(reviews_file)
reviews.createOrReplaceTempView('reviews')
reviews = reviews.select(['user_id', 'book_id', 'rating'])

# books_list = reviews.select('book_id').distinct().rdd.map(lambda r: r[0]).collect()

bookUserDf = reviews.groupBy('book_id').pivot('user_id').sum('rating').toPandas()
bookUserDf = bookUserDf.fillna(0)
