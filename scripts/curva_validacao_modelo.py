from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

symbol = 'PG'  
start_date = '2000-01-01'
end_date = '2025-01-31'

df = yf.download(symbol, start=start_date, end=end_date)

df = df[['Close']]
df.rename(columns={'Close': 'Price'}, inplace=True)
df['Date'] = df.index
df.reset_index(drop=True, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
df['Scaled_Price'] = scaler.fit_transform(df[['Price']])

model = load_model('models/lstm_stock_model.keras')

time_steps = 75 
dados_finais = df['Scaled_Price'].values[-(time_steps + 50):-50].reshape(1, time_steps, 1)

previsoes_reais = []

for i in range(50):
    previsao = model.predict(dados_finais)[0, 0]  
    previsoes_reais.append(previsao)  
    dados_finais = np.append(dados_finais[:, 1:, :], [[[df['Scaled_Price'].values[-50 + i]]]], axis=1)

previsoes_futuras = []
for _ in range(20):
    previsao = model.predict(dados_finais)[0, 0] 
    previsoes_futuras.append(previsao)
    dados_finais = np.append(dados_finais[:, 1:, :], [[[previsao]]], axis=1)

previsoes_reais_desnormalizadas = scaler.inverse_transform(np.array(previsoes_reais).reshape(-1, 1))
previsoes_futuras_desnormalizadas = scaler.inverse_transform(np.array(previsoes_futuras).reshape(-1, 1))

precos_reais_completos = np.concatenate((df['Price'].values[-504:-50], df['Price'].values[-50:]))

datas_reais = df['Date'].values[-504:]
datas_futuras = pd.date_range(start=datas_reais[-1], periods=21, freq='B')[1:] 

plt.figure(figsize=(12, 6))
plt.plot(datas_reais, precos_reais_completos, label='Preços Reais (Últimos 2 Anos)', color='blue')
plt.plot(datas_reais[-50:], previsoes_reais_desnormalizadas, label='Previsão (Últimos 50 dias)', color='orange')
plt.plot(datas_futuras, previsoes_futuras_desnormalizadas, label='Previsão (Próximos 20 dias)', color='red', linestyle='dashed')

plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('Comparação: Preços Reais vs Previsão (Últimos 2 Anos + Futuro)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)  
plt.show()
