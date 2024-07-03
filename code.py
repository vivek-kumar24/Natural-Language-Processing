#Solving lexical complexity problem usimg machine learning
#nlp task one

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Loading datasets
train_data = pd.read_csv(r"C:\Users\Vivek\Documents\lcp_single_train.tsv", sep='\t') 
test_data = pd.read_csv(r"C:\Users\Vivek\Downloads\lcp_single_test.tsv", sep='\t')


# Discretize the 'complexity' column
def discretize_complexity(df, n_bins=5, strategy='uniform'):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df['complexity_discrete'] = discretizer.fit_transform(df[['complexity']])
    return df

train_data = discretize_complexity(train_data)
test_data = discretize_complexity(test_data)

# Vectorizing the 'sentence' column using tfidf vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['sentence'])
X_test = vectorizer.transform(test_data['sentence'])

# Extract the labels
y_train = train_data['complexity_discrete']
y_test = test_data['complexity_discrete']

# Training classifiers
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[model_name] = {'accuracy': accuracy, 'f1_score': f1, 'model': model}
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    print("="*50)

# Plotting the results
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]
f1_scores = [results[model]['f1_score'] for model in model_names]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].barh(model_names, accuracies, color='skyblue')
ax[0].set_title('Model Accuracy')
ax[0].set_xlim(0, 1)
ax[0].set_xlabel('Accuracy')

ax[1].barh(model_names, f1_scores, color='lightgreen')
ax[1].set_title('Model F1 Score')
ax[1].set_xlim(0, 1)
ax[1].set_xlabel('F1 Score')

plt.tight_layout()
plt.show()

print(accuracies)
