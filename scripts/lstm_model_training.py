import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.losses import Loss
import tensorflow as tf
import joblib


# Trecho para contorle para verificar se a GPU estava sendo reconhecida corretamente com o cudnn/cuda
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detectada: {gpus[0]}")
else:
    print("GPU não detectada. O código será executado na CPU.")


# Aqui definimos o log que observavamos a evolução das perdas para entender se havia overfitting
class TrainingProgressLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Época {epoch + 1} concluída:")
        print(f" - Loss: {logs['loss']:.4f}")
        if 'val_loss' in logs:
            print(f" - Val Loss: {logs['val_loss']:.4f}")


# Escolhemos uma empresa com baixas variações, entendendo que o modelo simples seria mais interessante nesse caso
symbol = 'PG'  
start_date = '2000-01-01'
end_date = '2025-01-31'

# Coleta de dados do Yahoo Finance
df = yf.download(symbol, start=start_date, end=end_date)

# Tratamento dos dados da PG
df = df[['Close']]
df.rename(columns={'Close': 'Price'}, inplace=True)
df['Date'] = df.index
df['SMA_10'] = df['Price'].rolling(window=10).mean()
df.reset_index(drop=True, inplace=True)

print("Visualizando os dados antes da normalização:")
print(df.head())

# Normalização dos dados a serem usados no modelo utilizando o MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['Scaled_Price'] = scaler.fit_transform(df[['Price']])
print("Normalização concluída. Primeiros valores normalizados:")
print(df['Scaled_Price'].head())

# Salvando o scaler para recuperar na API de previsão
joblib.dump(scaler, 'scaler.pkl')

# Função para criação das sequências que serão usadas com base no time_Step escolhido
def create_sequences(data, time_steps):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)

# Configuração do número de time steps com base no DF normalizado
time_steps = 75
data = df['Scaled_Price'].values
x, y = create_sequences(data, time_steps)

# Dividir os dados, optamos por usar a referência de 80% para treino e 20% para teste
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Criação do modelo propriamente dito. Aqui foi ajustado inúmeras vezes para chegar em estatísticas boas sem overfitting
model = Sequential([
    LSTM(125, return_sequences=False, input_shape=(time_steps, 1)),
    Dropout(0.10),
    Dense(10),
    Dense(1)
])

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='huber')

# Definir o callback Early Stopping, visto que tivemos bastante problema com overfitting nas primeiras combinações de hiperparâmetros
early_stopping = EarlyStopping(
    monitor='val_loss',     
    patience=5,             
    restore_best_weights=True  
)

# Treinar o modelo
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=80,
    batch_size=16,
    callbacks=[early_stopping]
)

# Previsão
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
print(f"Previsões concluídas para {len(predictions)} amostras.")

# Desnormalizar os valores reais
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Cálculo das métricas estatísticas
mae = mean_absolute_error(real_prices, predictions)
rmse = np.sqrt(mean_squared_error(real_prices, predictions))
mape = np.mean(np.abs((real_prices - predictions) / real_prices)) * 100

real_prices = real_prices[~np.isnan(real_prices)]
predictions = predictions[~np.isnan(predictions)]

real_variations = np.diff(real_prices)

predicted_variations = np.diff(predictions)

# Logando as estatísticas reais para cruzar com previsões
real_stats = {
    "Média": np.mean(real_variations),
    "Desvio Padrão": np.std(real_variations),
    "Variação Mínima": np.min(real_variations),
    "Variação Máxima": np.max(real_variations),
}

# Logando as estatísticas das previsões
predicted_stats = {
    "Média": np.mean(predicted_variations),
    "Desvio Padrão": np.std(predicted_variations),
    "Variação Mínima": np.min(predicted_variations),
    "Variação Máxima": np.max(predicted_variations),
}

# Imprimir estatísticas
print("Estatísticas das Variações Reais:")
for k, v in real_stats.items():
    print(f"{k}: {v:.2f}")

print("\nEstatísticas das Variações Previstas:")
for k, v in predicted_stats.items():
    print(f"{k}: {v:.2f}")

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape:.2f}")

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(real_prices, label='Preços Reais')
plt.plot(predictions, label='Preços Previstos')
plt.title(f'{symbol} - Previsão de preço de ações')
plt.xlabel('Dias')
plt.ylabel('Preço')
plt.legend()
plt.show()

# Plotar a perda de treino vs validação para avaliar visualmente possibilidade de overfitting
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.title('Análise de Overfitting')
plt.show()

# Salvar o modelo para ser utilizado na API junto com o scaler
model.save('lstm_stock_model.keras')
