import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn as rnn_utils
import pandas as pd
import nltk

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Carregar o arquivo CSV
df = pd.read_csv('./src/csv/file.csv').head(5)

# Dividir o conjunto de dados em conjuntos de treinamento e teste
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Criar um dicionário de vocabulário e converter os textos em sequências de números
# Criar um dicionário de vocabulário
vocab = {}

# Iterar sobre as linhas do conjunto de treinamento e converter os textos em sequências de números
sequences = []

for i, row in train_df.iterrows():
    tokens = nltk.word_tokenize(row['Body'].lower())
    sequence = [vocab.setdefault(token, len(vocab)) for token in tokens]
    sequences.append(sequence)

# Preencher as sequências com zeros para que todas tenham o mesmo comprimento
train_padded_sequences = rnn_utils.pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)
train_inputs = train_padded_sequences[:, :80].to(device)
train_labels = torch.tensor(train_df['Score'].values).unsqueeze(1).to(device)
train_labels_normalized = train_labels / train_labels.max()

# Iterar sobre as linhas do conjunto de teste e converter os textos em sequências de números
test_sequences = []
for i, row in test_df.iterrows():
    tokens = nltk.word_tokenize(row['Body'].lower())
    sequence = [vocab.get(token, 0) for token in tokens]
    test_sequences.append(sequence)

# Preencher as sequências com zeros para que todas tenham o mesmo comprimento
test_padded_sequences = rnn_utils.pad_sequence([torch.tensor(seq) for seq in test_sequences], batch_first=True)
test_inputs = test_padded_sequences[:, :80].to(device)
test_labels = torch.tensor(test_df['Score'].values).unsqueeze(1).to(device)
test_labels_normalized = test_labels / test_labels.max()

seqs = []
for i, row in df.iterrows():
    tokens = nltk.word_tokenize(row['Body'].lower())
    seq = [vocab.get(token, 0) for token in tokens]
    seqs.append(seq)

padded_seqs = rnn_utils.pad_sequence([torch.tensor(seq) for seq in seqs], batch_first=True)

# Criar um modelo de classificação de texto com uma camada de embedding, uma camada LSTM e uma camada linear
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, (h_n, c_n) = self.lstm(embedded)
        linear_output = self.linear(lstm_output[:, -1, :])  # Use a última saída oculta do LSTM
        return linear_output.squeeze()

# Definir hiperparâmetros do modelo
vocab_size = len(vocab)
embedding_dim = 50
hidden_dim = 100
learning_rate = 0.001
num_epochs = 300

# Criar uma instância do modelo e definir a função de perda e o otimizador
model = TextClassifier(vocab_size, embedding_dim, hidden_dim).to(device)
# model = torch.load('./src/models/model01.pt')

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treinar o modelo
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_inputs)

    loss = criterion(outputs.squeeze(), train_labels_normalized.float().squeeze())
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Época {epoch}, perda: {loss.item()}")

# Avaliar o modelo no conjunto de teste
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_probs = torch.sigmoid(test_outputs)
    test_preds = (test_probs > 0.5).long().cpu()
    accuracy = (test_preds == test_labels).sum().float() / len(test_labels)
    print(f"Acurácia no conjunto de teste: {accuracy.item()}")
    # print("Probabilidades brutas do exemplo de teste:", test_probs.squeeze().tolist())

def predict(text, model, vocab, max_length):
    tokens = nltk.word_tokenize(text.lower())
    sequence = [vocab.get(token, 0) for token in tokens]
    padded_sequence = torch.zeros(1, max_length).long()
    sequence_length = min(len(sequence), max_length)
    padded_sequence[0, :sequence_length] = torch.tensor(sequence[:sequence_length])
    
    with torch.no_grad():
        output = model(padded_sequence)
        prob = torch.sigmoid(output)
        prediction = (prob > 0.5).long()
        
    print(f"Forma do tensor 'prediction': {prediction.shape}")  # Adicione esta linha
    return prediction.item()

for i in range(len(df)):
    seq = padded_seqs[i:i+1]
    output = model(seq)
    prediction = torch.round(torch.sigmoid(output)).item()

    print(f"Pergunta: {df['Body'][i]}\nClassificação: {prediction}\n")