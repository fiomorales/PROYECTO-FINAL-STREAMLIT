
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo entrenado
gb_optimized = joblib.load('modelo_gradient_boosting.pkl')  # Reemplaza con la ruta de tu modelo

# Interfaz de usuario de Streamlit
st.title("Predicción de Inspección Aduanera (Rojo o Verde)")

st.write("Introduce los valores de las características para realizar la predicción:")

# Crear campos de entrada para las 5 características principales
tipo_cambio = st.number_input("Tipo de Cambio", min_value=0.0, step=0.01)
peso_bruto = st.number_input("Peso Bruto de la Mercancía", min_value=0.0, step=0.01)
total_fletes = st.number_input("Total Fletes", min_value=0.0, step=1.0)
pedimento = st.number_input("Pedimento", min_value=0, step=1)
valor_aduana = st.number_input("Valor en Aduana", min_value=0.0, step=1.0)

# Botón para realizar la predicción
if st.button("Predecir"):
    # Crear un DataFrame de entrada para el modelo
    datos_entrada = pd.DataFrame({
        'PesoBrutoMercancia': [peso_bruto],
        'TipoCambio': [tipo_cambio],
        'ImpuestosPagados': [0],  # Usa un valor predeterminado si no tienes el dato
        'SeccionAduanera': [0],   # Usa un valor predeterminado si no tienes el dato
        'Pedimento': [pedimento],
        'TotalFletes': [total_fletes],
        'PrecioUnitario': [0],    # Usa un valor predeterminado si no tienes el dato
        'ValorAduana': [valor_aduana]
    })

    # Realizar la predicción
    prediccion = gb_optimized.predict(datos_entrada)
    resultado = "Verde" if prediccion[0] == 1 else "Rojo"
    
    st.write("Resultado de la predicción:", resultado)
