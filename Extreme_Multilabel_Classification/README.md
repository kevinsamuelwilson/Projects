# Extreme Multilabel Classification

#### May 2020 - NYU Center for Data Science
#### Key learnings: machine learning, multi-layer perceptron, multilabel classification, model selection

[project report](Extreme_Multilabel_Classification_Report.pdf)

The extreme classification setting involves problems in which we hope to assign multiple labels to items from a large set of candidate labels. We were tasked with predicting labels for a dataset of European legal documents containing roughly 15,000 examples with about 4,000 labels applied at a rate of roughly 25 examples per label. Our version of the dataset has the data preprocessed so that the 5,000 features are TF-IDF scores of words from the original text and the labels are transformed into integers. We explored and compared a wide set of models including multi-layer perceptron, ML-KNN, radius neighbors, random forest, classifier chains, FastXML, and CraftML (an algorithm specifically designed for extreme classification problems that reduces the dimensionality of the feature and label spaces using a series of random projections). The CraftML model scored highest on test data on our primary evaluation metric, LRAP (Label Ranking Average Precision).
