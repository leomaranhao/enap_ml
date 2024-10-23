import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Carregando o tokenizer e o modelo pré-treinado
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_bert_embeddings(texts, model, tokenizer, max_length=128):
    embeddings = []
    for text in texts:
        # Tokenização
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        
        # Gerar embeddings com o modelo BERT
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Pegamos o vetor do CLS token como embedding da frase
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    
    return embeddings

texts = [
    "Este é um exemplo de texto.",
    "Outro exemplo de texto para classificar.",
    "Mais um exemplo para treinar o modelo."
]
labels = ["classe1", "classe2", "classe1"]

# Codificação dos rótulos (transformar texto em números)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

embeddings = get_bert_embeddings(texts, model, tokenizer)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Treinar o modelo SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = sgd_clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")
