import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

symbol = 'PG'  
start_date = '2000-01-01'
end_date = '2025-01-31'

# Coleta de dados do Yahoo Finance
df = yf.download(symbol, start=start_date, end=end_date)

df = df[['Close']]
df.rename(columns={'Close': 'Price'}, inplace=True)
df['Date'] = df.index
df['SMA_10'] = df['Price'].rolling(window=10).mean()
df.reset_index(drop=True, inplace=True)

print("Visualizando os dados antes da normalização:")
print(df.head())

# Verificar o tipo de dados
print("Tipos de dados no DataFrame:")
print(df.dtypes)

# Verificar se há NaN ou valores inválidos
print("Verificando NaN nos dados:")
print(df.isnull().sum())

print(f"Tamanho do DataFrame: {df.shape}")

# Plotar os dados históricos reais
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label=f'{symbol} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{symbol} Historical Prices')
plt.legend()
plt.show()
