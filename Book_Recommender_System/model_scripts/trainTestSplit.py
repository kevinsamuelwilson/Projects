'''
Usage:
    $ spark-submit trainTestSplit.py in_path out_path_start

    e.g.
    $ spark-submit trainTestSplit.py 'reviews_full.parquet' 'reviews'
'''
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession


def train_val_test_split(spark, in_path, out_path_start, downsample=False):
    '''
    takes all reviews and splits them into train, validation, and test
    using the guidelines in the read me file
    then, writes each split to a parquet file

    spark - the spark session
    in_path - the filepath for the incoming parquet data
    out_path_start -    the output file path beginning
                        each output parquet file will end with either
                        '_val.parquet', '_test.parquet', '_train.parquet'
                        depending on the dataset
    '''
    all_reviews = spark.read.parquet(in_path)
    all_reviews.createOrReplaceTempView('all_reviews')


    # get list of unique users to split (only taking users with 10 or more reviews)
    # NB changed to 'book_id' as no 'review_id' in dataset given
    unique_users = spark.sql('SELECT user_id FROM all_reviews GROUP BY user_id HAVING COUNT(book_id) >= 10')
    unique_users.createOrReplaceTempView('unique_users')


    # split unique_users into train, val, and test using randomSplit
    train,val,test = unique_users.randomSplit([0.6, 0.2, 0.2], seed = 2020)

    if downsample==True:
        unique_users_small = unique_users.sample(False, 0.01, seed=2020)
        train,val,test = unique_users_small.randomSplit([0.6, 0.2, 0.2], seed = 2020)
        out_path_start = out_path_start + '_small'

    # join user_id lists with review data for each split
    # and select only the columns we care about
    train_reviews = all_reviews.join(train, how = 'right', on = 'user_id').select('user_id', 'book_id', 'is_read', 'rating')
    val_reviews = all_reviews.join(val, how = 'right', on = 'user_id').select('user_id', 'book_id', 'is_read', 'rating')
    test_reviews = all_reviews.join(test, how = 'right', on = 'user_id').select('user_id', 'book_id', 'is_read', 'rating')

    # split val by dividing each users reviews in half using splitBY()
    frac =  dict((e.user_id, 0.5) for e in val.collect())
    val_reviews_in_train = val_reviews.sampleBy('user_id', frac)
    val_reviews_in_val = val_reviews.subtract(val_reviews_in_train)
    val_reviews_in_val.repartition('user_id').write.parquet(out_path_start + '_val.parquet')

    # split test by dividing each users reviews in half using splitBY()
    frac =  dict((e.user_id, 0.5) for e in test.collect())
    test_reviews_in_train = test_reviews.sampleBy('user_id', frac)
    test_reviews_in_test = test_reviews.subtract(test_reviews_in_train)
    test_reviews_in_test.repartition('user_id').write.parquet(out_path_start + '_test.parquet')

    # append train data
    train_reviews = train_reviews.union(val_reviews_in_train).union(test_reviews_in_test)
    train_reviews.repartition('user_id').write.parquet(out_path_start + '_train.parquet')


# Enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('spark-goodreads-recsys').getOrCreate()


    # Get in_path and out_path_start from command line
    in_path = sys.argv[1]
    # in_path = 'reviews_full.parquet'

    out_path_start = sys.argv[2]
    # out_path_start ='reviews'

    try:
        downsample = sys.argv[3]
    except:
        downsample = False

    # Call data loading routine
    train_val_test_split(spark, in_path, out_path_start)
