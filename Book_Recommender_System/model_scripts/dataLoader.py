'''
Usage:
Within the pyspark terminal, use

>>> import dataLoader as dl
>>> # load the initial dataset
>>> dl.load_data(filetype, spark, in_path, out_path)
>>> # split into train, test, and validation and save to hdfs
>>> dl.train_val_test_split(spark, in_path, out_path_start, downsample)

'''


def load_data (filetype, spark, in_path, out_path):
    '''
    loads the large csv data set into spark,
    filters out the unread and unreviewed rows
    and saves to a parquet file
    
    filetype - the input file type (csv, json)
    spark - the spark session
    in_path - the input file path of the data
    out_path - the output file path of the data 
    '''
    
    if (filetype == 'json'):
        df = spark.read.json(in_path)
        #cast rating to a double
        df = df.withColumn('rating', df['rating'].cast('double'))
    
    if (filetype == 'csv'):
        df = spark.read.csv(in_path, header=True, schema='user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
    
    df.createOrReplaceTempView('df')
    df = df.filter(df.is_read == True)
    df = df.filter(df.rating != 0)
    
    # write to parquet
    df.repartition('user_id').write.parquet(out_path)
    
def train_val_test_split(spark, in_path, out_path_start, downsample = False):
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
    
    # get list of books that appear more than once
    books_more_than_one = spark.sql('SELECT book_id FROM all_reviews GROUP BY book_id HAVING COUNT(book_id) > 2')
    all_reviews = all_reviews.join(books_more_than_one, how = 'inner', on = 'book_id')
    all_reviews.createOrReplaceTempView('all_reviews')
    
    # get list of unique users to split (only taking users with 10 or more reviews)
    unique_users = spark.sql('SELECT user_id FROM all_reviews GROUP BY user_id HAVING COUNT(book_id) >= 10')
    unique_users.createOrReplaceTempView('unique_users')
    
    # split unique_users into train, val, and test using randomSplit
    
    if downsample:
        unique_users = unique_users.sample(False, 0.5, seed = 2020)
        out_path_start += '_small'
    
    train,val,test = unique_users.randomSplit([0.6, 0.2, 0.2], seed = 2020)
    
    # join user_id lists with review data for each split
    # and select only the columns we care about
    train_reviews = all_reviews.join(train, how = 'right', on = 'user_id').select('user_id', 'book_id', 'rating')
    val_reviews = all_reviews.join(val, how = 'right', on = 'user_id').select('user_id', 'book_id', 'rating')
    test_reviews = all_reviews.join(test, how = 'right', on = 'user_id').select('user_id', 'book_id', 'rating')
    
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
    train_reviews = train_reviews.union(test_reviews_in_train).union(val_reviews_in_train)
    train_reviews.repartition('user_id').write.parquet(out_path_start + '_train.parquet')
