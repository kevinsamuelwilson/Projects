# Goodreads Book Recommender System

#### May 2020 - NYU Center for Data Science
#### Key learnings: Hadoop, Spark, PySpark, big data management, machine learning, recommender systems

[paper](Goodreads_Report.pdf)

We developed a recommender system for books using data obtained from the book review website Goodreads in 2017. Our system was built using the alternating least squares (ALS) model in Spark to learn latent factor representations of users and books. We trained our model on a subset of the data and used holdout validation data to tune and evaluate different model hyper-parameters. We considered various metrics for evaluating performance for the top 500 book recommendations for each user including precision at 500, mean average precision, and others, and examined our final model’s performance on test data not previously seen. Finally, we implemented an extension to visualize the latent item factors using t-SNE to illustrate how the items are distributed in the learned space.
The dataset included approximately 223 million individual book ratings, across 2.4 million unique books and 876,000 unique users.
