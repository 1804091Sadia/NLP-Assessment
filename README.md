# Twitter Sentiment Analysis (NLP Assessment)

A **Natural Language Processing (NLP) project** for **tweet categorization**. The goal is to classify tweets about a specific entity into **Positive, Negative, or Neutral** sentiments using various machine learning algorithms.

---

## Project Description

This project uses **entity-level sentiment analysis** to categorize tweets based on the sentiment expressed toward a given entity. Tweets not relevant to the entity are considered **Neutral**. The project involves:

* Data cleaning and preprocessing
* Tokenization and vectorization using **TF-IDF**
* Training and evaluating multiple machine learning models
* Hyperparameter tuning to optimize model performance
* Visualization of results to select the **best-performing model**

---

## Dataset

* **Training Set:** `twitter_training.csv`
* **Validation Set:** `twitter_validation.csv`

Each dataset contains:

* `tweet` – the text of the tweet
* `entity` – the target entity
* `sentiment` – the sentiment label (`Positive`, `Negative`, `Neutral`)

---

## Methods

The following machine learning algorithms are used for classification:

1. **Support Vector Machine (SVM)**
2. **Decision Tree**
3. **Random Forest**
4. **Multinomial Naive Bayes**

Steps include:

* Cleaning and preprocessing text (removing punctuation, lowercasing, stopwords, etc.)
* Tokenizing text using **NLTK**
* Vectorizing tweets using **TF-IDF**
* Hyperparameter tuning for each model
* Evaluating model performance on the validation set
* Visualizing results (accuracy, confusion matrix, etc.)

---

## Usage

1. Upload datasets in Google Colab or Jupyter Notebook.
2. Preprocess the data:

```python
import pandas as pd
train_df = pd.read_csv("twitter_training.csv")
val_df = pd.read_csv("twitter_validation.csv")
```

3. Apply preprocessing, tokenization, and TF-IDF vectorization.
4. Train and evaluate models. Example using SVM:

```python
from sklearn.svm import SVC
model = SVC(C=1.0, kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

5. Visualize evaluation metrics and select the **best algorithm**.

---

## Libraries Required

* `pandas`
* `numpy`
* `nltk`
* `scikit-learn`
* `matplotlib` / `seaborn`

Install via pip if needed:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

---

## Goal

* Identify the **best-performing ML model** for tweet sentiment classification
* Provide insights from model evaluation and visualization

