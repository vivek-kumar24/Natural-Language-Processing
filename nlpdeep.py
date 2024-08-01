#!/usr/bin/env python
# coding: utf-8

# In[17]:


#1st cnn models used to perform the task
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_data = pd.read_csv(r"C:\Users\Vivek\Documents\lcp_single_train.tsv", sep='\t')
test_data = pd.read_csv(r"C:\Users\Vivek\Documents\lcp_single_test.tsv", sep='\t')

# Discretize the 'complexity' column
def discretize_complexity(df, n_bins=5, strategy='uniform'):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df['complexity_discrete'] = discretizer.fit_transform(df[['complexity']]).astype(int)
    return df

train_data = discretize_complexity(train_data)
test_data = discretize_complexity(test_data)

# Preprocess the 'sentence' column using Tokenizer and pad_sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128

def tokenize_sentences(sentences):
    return tokenizer(sentences.tolist(), padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")

train_encodings = tokenize_sentences(train_data['sentence'])
test_encodings = tokenize_sentences(test_data['sentence'])

# Create PyTorch Datasets
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are of type Long
        return item

    def __len__(self):
        return len(self.labels)

train_labels = train_data['complexity_discrete'].values
test_labels = test_data['complexity_discrete'].values

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the CNN model using PyTorch
class CNNModel(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.conv1 = nn.Conv1d(64, 64, 5, padding=2)
        self.pool = nn.MaxPool1d(4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Define the model
vocab_size = tokenizer.vocab_size
num_classes = len(np.unique(train_labels))

model = CNNModel(vocab_size, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(classification_report(all_labels, all_preds))
    return accuracy, f1

# Evaluate the model
cnn_accuracy, cnn_f1 = evaluate_model(model, test_loader)

# Plotting the results
model_names = ['CNN']
accuracies = [cnn_accuracy]
f1_scores = [cnn_f1]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

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


# In[21]:


#2nd rnn models to perform the task
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_data = pd.read_csv(r"C:\Users\Vivek\Documents\lcp_single_train.tsv", sep='\t')
test_data = pd.read_csv(r"C:\Users\Vivek\Documents\lcp_single_test.tsv", sep='\t')

# Discretize the 'complexity' column
def discretize_complexity(df, n_bins=5, strategy='uniform'):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df['complexity_discrete'] = discretizer.fit_transform(df[['complexity']]).astype(int)
    return df

train_data = discretize_complexity(train_data)
test_data = discretize_complexity(test_data)

# Preprocess the 'sentence' column using Tokenizer and pad_sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128

def tokenize_sentences(sentences):
    return tokenizer(sentences.tolist(), padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")

train_encodings = tokenize_sentences(train_data['sentence'])
test_encodings = tokenize_sentences(test_data['sentence'])

# Create PyTorch Datasets
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are of type Long
        return item

    def __len__(self):
        return len(self.labels)

train_labels = train_data['complexity_discrete'].values
test_labels = test_data['complexity_discrete'].values

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the RNN model using PyTorch
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, hn = self.rnn(x)
        hn = hn.squeeze(0)
        x = self.fc(hn)
        return self.softmax(x)

# Define the model
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 64
output_dim = len(np.unique(train_labels))

model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 12

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(classification_report(all_labels, all_preds))
    return accuracy, f1

# Evaluate the model
rnn_accuracy, rnn_f1 = evaluate_model(model, test_loader)

# Plotting the results
model_names = ['RNN']
accuracies = [rnn_accuracy]
f1_scores = [rnn_f1]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

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


# In[20]:


#3rd ffnn models used to perform the task
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_data = pd.read_csv(r"C:\Users\Vivek\Documents\lcp_single_train.tsv", sep='\t')
test_data = pd.read_csv(r"C:\Users\Vivek\Documents\lcp_single_test.tsv", sep='\t')

# Discretize the 'complexity' column
def discretize_complexity(df, n_bins=5, strategy='uniform'):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df['complexity_discrete'] = discretizer.fit_transform(df[['complexity']]).astype(int)
    return df

train_data = discretize_complexity(train_data)
test_data = discretize_complexity(test_data)

# Preprocess the 'sentence' column using Tokenizer and pad_sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128

def tokenize_sentences(sentences):
    return tokenizer(sentences.tolist(), padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")

train_encodings = tokenize_sentences(train_data['sentence'])
test_encodings = tokenize_sentences(test_data['sentence'])

# Create PyTorch Datasets
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are of type Long
        return item

    def __len__(self):
        return len(self.labels)

train_labels = train_data['complexity_discrete'].values
test_labels = test_data['complexity_discrete'].values

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the FFNN model using PyTorch
class FFNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(FFNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * max_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.view(x.size(0), -1)  # Flatten the embeddings
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Define the model
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 64
output_dim = len(np.unique(train_labels))

model = FFNNModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 12

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(classification_report(all_labels, all_preds))
    return accuracy, f1

# Evaluate the model
ffnn_accuracy, ffnn_f1 = evaluate_model(model, test_loader)

# Plotting the results
model_names = ['FFNN']
accuracies = [ffnn_accuracy]
f1_scores = [ffnn_f1]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

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


# In[ ]:




