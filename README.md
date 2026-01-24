# SMS Spam Detection & Deployment with Streamlit

This project implements an end-to-end Machine Learning solution to classify SMS messages as either **Spam** or **Ham** (legitimate). It covers data preprocessing, statistical modeling using Naive Bayes, and the creation of a web interface using **Streamlit** for real-time predictions.

## 📌 Project Overview

Spam detection is a classic NLP (Natural Language Processing) problem. This project uses a dataset of over 5,500 messages to train a model that can distinguish between personal communication and unsolicited advertisements or scams.

## 📊 Dataset Analysis

The model is trained on a tab-separated dataset (`sms.csv`) with two labels:

* **Ham (1)**: Legitimate messages (approx. 86.6% of the data).
* **Spam (0)**: Junk or phishing messages (approx. 13.4% of the data).

### Feature Engineering

A significant feature analyzed in this project is **Message Length**. Typically, spam messages tend to be longer and more "urgent" in nature compared to concise ham messages.

## 🛠️ Tech Stack

* **Language**: Python
* **Data Handling**: Pandas, NumPy
* **NLP & ML**: Scikit-Learn (`TfidfVectorizer`, `MultinomialNB`)
* **Visualization**: Matplotlib, Seaborn
* **Web Framework**: [Streamlit](https://streamlit.io/)
* **Model Persistence**: Pickle

## 🚀 Machine Learning Pipeline

1. **Preprocessing**: Text is cleaned and converted to numerical format using **TF-IDF Vectorization**, which weighs words based on their importance across the dataset.
2. **Training**: A **Multinomial Naive Bayes** classifier is used, which is highly effective for discrete features like word counts.
3. **Evaluation**: The model achieves an **Accuracy of ~95.8%**.
* **Precision (Spam)**: 0.99 (Very few legitimate messages are misclassified as spam).
* **Recall (Spam)**: 0.73 (The model identifies most, but not all, spam messages).



## 🌐 Streamlit Web Application

The project includes a dedicated app directory (`spam_detector_app`) containing `app.py`. This script creates a user-friendly interface where anyone can paste an SMS and get an instant prediction.

### How to Run the App

1. Navigate to the directory:
```bash
cd spam_detector_app

```


2. Launch the Streamlit server:
```bash
streamlit run app.py

```



## 📦 Saved Assets

To ensure the app works without retraining, the following files are exported via Pickle:

* `vectorizer.pkl`: The fitted TF-IDF transformer.
* `spam_model.pkl`: The trained MultinomialNB model.

---

**Example Prediction:**

> *Input*: "Congratulations! You've won a free prize. Click here to claim."
> *Result*: **Spam**
