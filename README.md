# Fake-News-Detection Using Machine Learning- An NLP Approach
## Research Question
“How can machine learning techniques be used to accurately classify news articles as real or fake based on textual features?”
This project implements fake news detection using machine learning and deep learning models. 

![image alt](https://github.com/amaluvincent/Fake-News-Detection/blob/main/intro%20image.jpg?raw=true)

The dataset used is the ISOT Fake News Dataset, which contains labeled news articles categorized as "True" (0) or "Fake" (1). The goal is to accurately classify news articles and evaluate model performance using standard metrics.
## Dataset
The ISOT Fake News Dataset, curated by Dr. Ahmed H. Ahmed at the University of Victoria, consists of two types of news articles:

* True News: Obtained from Reuters.
* Fake News: Collected from various unreliable sources flagged by fact-checking organizations.
You can learn more about the dataset from here: https://www.impactcybertrust.org/dataset_view?idDataset=952 

The dataset contains:

* Class 0 (True News): Reliable and fact-checked articles.
* Class 1 (Fake News): Articles flagged as unreliable or misleading.
## Features
* Data Preprocessing such as cleaning, tokenization, and preparation of the dataset.
* Model Implementation:
  1. Logistic Regression (LR)
  2. Random Forest (RF)
  3. Naive Bayes (NB)
  4. Long Short-Term Memory (LSTM) Neural Network
* Performance Metrics: Accuracy, Precision, Recall, and F1-score are used to evaluate each model.
## Results
Out of all these models Logitic regression and LSTM is the best. 
The barplot below summarizes the performance of the models:
![image alt](https://github.com/amaluvincent/Fake-News-Detection/blob/main/result%20comparison.png?raw=true)
## Key Files
* data_pre_processing_v2.ipynb: Script for cleaning and tokenizing the text data.
* Fake_news_detection_NLP_approach_V7.ipynb: Contains the implementation of Logistic Regression, Random Forest, and Naive Bayes.
* Fake_news_detection_NLP_approach_V8.ipynb: Implementation of the LSTM neural network.
* FINAL_CODE.ipynb: End-to-end script to preprocess the data, train models, and evaluate performance.
## Dependencies
* Python 3.8 or later
* Libraries:
     * numpy
     * pandas
     * scikit-learn
     * nltk
     * tensorflow (for LSTM)
## Installation
Install required dependencies:
* pip install -r requirements.txt
* pip install pandas numpy matplotlib seaborn scikit-learn nltk
* pip install tensorflow keras

## Future Work
* Experiment with additional deep learning architectures, such as Transformers.
* Use more advanced text embeddings, such as BERT or Word2Vec.
* Explore additional feature engineering techniques.
## Author
Amalu Vincent



