from fastapi import FastAPI, HTTPException
import uvicorn
import tensorflow as tf
import numpy as np
import joblib
import logging
import time
import tracemalloc  

# Configuração do logging
logging.basicConfig(
    filename='model_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

app = FastAPI()

# Carregar o arquivo do modelo e o scaler utilizado no treinamento
model = tf.keras.models.load_model('lstm_stock_model.h5')
scaler = joblib.load('scaler.pkl')

# Time steps utilizados no treinamento
time_steps = 75

@app.post('/predict')
def predict(prices: list[float]):
    if len(prices) != time_steps:
        raise HTTPException(status_code=400, detail=f"Esperados {time_steps} valores, mas recebidos {len(prices)}.")
    
    try:
        # Iniciar o monitoramento de memória e tempo de CPU
        tracemalloc.start()
        cpu_start_time = time.process_time()

        # Normalizar os dados de entrada
        scaled_data = scaler.transform(np.array(prices).reshape(-1, 1))
        x_input = np.array([scaled_data[-time_steps:]])

        # Previsão
        prediction = model.predict(x_input)
        predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

        # Finalizar o monitoramento de CPU e memória
        cpu_time_taken = time.process_time() - cpu_start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Converter tudo para megabytes
        current_memory = current / (1024 * 1024)
        peak_memory = peak / (1024 * 1024)

        # Logar tudo no arquivo de logs
        logging.info(f"Tempo de CPU (s): {cpu_time_taken:.4f} - Memória Atual (MB): {current_memory:.2f} - Memória de Pico (MB): {peak_memory:.2f}")

        return {"predicted_price": float(predicted_price)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
