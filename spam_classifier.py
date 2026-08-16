# Task 1: Spam Email/SMS Classifier

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the dataset
# Download "SMSSpamCollection" from:
# https://archive.ics.uci.edu/dataset/228/sms+spam+collection

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("First 5 rows:")
print(data.head())


# 2. Convert labels into numbers
# ham = 0
# spam = 1

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# 3. Separate messages and labels

X = data["message"]
y = data["label"]


# 4. Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Convert text into numerical features using TF-IDF

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 6. Create Naive Bayes classifier

model = MultinomialNB()


# 7. Train the model

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# 8. Make predictions

y_pred = model.predict(X_test_tfidf)


# 9. Check accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")


# 10. Classification report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


# 11. Function to predict a new message

def predict_message(message):

    message_tfidf = vectorizer.transform([message])

    prediction = model.predict(message_tfidf)[0]

    probability = model.predict_proba(message_tfidf)[0]

    if prediction == 1:
        result = "SPAM"
        confidence = probability[1] * 100
    else:
        result = "HAM"
        confidence = probability[0] * 100

    print("\nMessage:", message)
    print("Prediction:", result)
    print("Confidence:", round(confidence, 2), "%")


# 12. Test some messages

predict_message(
    "Congratulations! You have won a free prize. Call now to claim!"
)

predict_message(
    "Hi, are you coming to college today?"
)

predict_message(
    "URGENT! You won 50000 dollars. Claim your prize now!"
)

predict_message(
    "Please send me the notes after class."
)


# 13. Enter your own message

print("\n-------------------------------")
print("Enter your own message")
print("Type 'exit' to stop")
print("-------------------------------")

while True:

    message = input("\nEnter message: ")

    if message.lower() == "exit":
        break

    predict_message(message)

