import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Carregar os Arquivos
print("Carregando datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Pré-processamento e Definição dos Dados de Treino
# Usamos o arquivo 'train.csv' INTEIRO para treino
X_train = train_df.drop(['id', 'target'], axis=1)
y_train = train_df['target']

# Codificar a variável alvo (converter texto para números)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
print(f"Classes treinadas: {le.classes_}")

# 3. Definição dos Dados de Teste
# O arquivo test.csv deve ter as mesmas colunas de features, sem o 'target'
X_test = test_df.drop(['id'], axis=1)

# 4. Criação e Treinamento do Modelo
print("Treinando o modelo com TODO o conjunto de dados...")
# n_jobs=-1 usa todos os núcleos do processador para ser mais rápido
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Aqui acontece a mágica: passamos X_train e y_train completos
clf.fit(X_train, y_train_encoded)

# 5. Previsão Final
print("Gerando previsões para o arquivo de teste...")
test_preds_encoded = clf.predict(X_test)

# Converter previsões numéricas de volta para 'attack'/'normal'
test_preds_labels = le.inverse_transform(test_preds_encoded)

# 6. Salvar Arquivo de Submissão
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': test_preds_labels
})

submission.to_csv('submission_full_train.csv', index=False)
print("Sucesso! Arquivo 'submission_full_train.csv' criado.")