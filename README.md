# Spotify Review Sentiment Analysis

## Overview

This project focuses on sentiment analysis of Spotify user reviews. The goal is to preprocess the text data, extract features, build models, and evaluate their performance in predicting the sentiment score of the reviews.

## Project Structure

- **Data Reading:** The project begins by loading and inspecting the Spotify reviews dataset.
  
- **Data Preprocessing:** 
  - **Text Cleaning:** Lowercasing, removing punctuation, and filtering stop words.
  - **Lemmatization and Stemming:** Reducing words to their base forms.
  
- **Feature Extraction:**
  - **TF-IDF Vectorization:** Transforming text data into numerical features.
  - **Tokenization and Padding:** Converting text into sequences and padding them to ensure uniform input size for the neural network.

- **Modeling:**
  - **Naive Bayes & Random Forest:** Implementing traditional machine learning models to establish baselines.
  - **Neural Network:** Developing a custom sequential neural network for sentiment prediction.

- **Model Evaluation:**
  - **Accuracy & Loss Metrics:** Tracking the model's performance across training epochs.
  - **Visualization:** Plotting accuracy and loss to understand model behavior.

## Installation

To run this project, you'll need to install the following dependencies:

```bash
pip install pandas numpy nltk scikit-learn tensorflow matplotlib
```

## Usage

### Data Preprocessing

1. **Load the Dataset:**
   ```python
   data = pd.read_csv('spotify_reviews.csv')
   ```

2. **Clean and Preprocess Text:**
   ```python
   def preprocessing(text):
       # Add your text preprocessing steps here
       return processed_text

   data['content'] = data['content'].apply(preprocessing)
   ```

3. **Feature Extraction:**
   ```python
   tfidf = TfidfVectorizer()
   new_x = tfidf.fit_transform(data['content'])
   ```

### Model Building

1. **Train Naive Bayes Model:**
   ```python
   model_nb = MultinomialNB()
   model_nb.fit(x_train, y_train)
   ```

2. **Train Random Forest Model:**
   ```python
   model_rf = RandomForestClassifier()
   model_rf.fit(x_train, y_train)
   ```

3. **Build and Train Neural Network:**
   ```python
   model = k.models.Sequential([
       k.layers.Embedding(vocabulary_size, 100, input_length=maxlen),
       k.layers.GlobalAveragePooling1D(),
       k.layers.Dense(128, activation="relu"),
       k.layers.Dense(24, activation="relu"),
       k.layers.Dense(5, activation="softmax")
   ])
   model.compile(optimizer="adam", loss=k.losses.CategoricalCrossentropy(), metrics=["accuracy"])
   history = model.fit(x_train1, y_train1, epochs=10, validation_data=(x_test1, y_test1), verbose=2)
   ```

### Evaluation

Evaluate the model's performance and visualize the results:

```python
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Epochs")
plt.legend()
plt.grid()

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss over Epochs")
plt.legend()
plt.grid()
```

## Conclusion

This project highlights the application of natural language processing and machine learning techniques for sentiment analysis. By comparing traditional machine learning models with a custom neural network, valuable insights into model performance were gained, particularly in the context of text data.

## Future Work

Possible future directions include fine-tuning hyperparameters, experimenting with more complex neural network architectures, and applying the model to other text datasets.
