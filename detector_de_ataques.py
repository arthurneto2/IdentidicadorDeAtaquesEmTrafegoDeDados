import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier

# 1. Carregar os Arquivos
print("Carregando datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Preparar Dados de Treino
X_train = train_df.drop(['id', 'target'], axis=1)
y_train = train_df['target']

# Codificar o alvo
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# 3. Preparar Dados de Teste
X_test = test_df.drop(['id'], axis=1)

# 4. Definir os Modelos do Ensemble
print("Configurando o Ensemble...")

# Modelo 1: Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=300, 
    random_state=42, 
    n_jobs=-1
)

# Modelo 2: Histogram-based Gradient Boosting (Estado da arte para dados tabulares no sklearn)
hgb_clf = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=200,          # Mais iterações para aprender detalhes finos
    random_state=42
)

# Combinando os dois com "Soft Voting" (Média das probabilidades)
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('hgb', hgb_clf)],
    voting='soft',
    n_jobs=-1
)

# 5. Treinamento
print("Treinando o Ensemble...")
voting_clf.fit(X_train, y_train_encoded)

# 6. Previsão
print("Gerando previsões...")
test_preds_encoded = voting_clf.predict(X_test)
test_preds_labels = le.inverse_transform(test_preds_encoded)

# 7. Salvar
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': test_preds_labels
})

submission.to_csv('submission_ensemble.csv', index=False)
print("Sucesso! Arquivo 'submission_ensemble.csv' gerado.")