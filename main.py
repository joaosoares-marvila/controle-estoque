import os
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.linear_model import LinearRegression

def train_and_predict(csv_file: str, product_id: str) -> dict:

    df_product_entries = pd.read_csv(csv_file, sep=',')

    # Extrair ano e mês da coluna 'entry_at'
    df_product_entries['entry_at'] = pd.to_datetime(df_product_entries['entry_at'])
    df_product_entries['year_month'] = df_product_entries['entry_at'].dt.to_period('M')

    # Agrupar os dados utilizando com referência 'product_id', 'name' e 'year_month' e somar as quantidades
    monthly_data = df_product_entries.groupby(['product_id', 'name', 'year_month'])['quantity'].sum().reset_index()

    # Converte em datetime
    monthly_data['year_month'] = monthly_data['year_month'].astype(str)
    monthly_data['year_month'] = pd.to_datetime(monthly_data['year_month'])

    # Criar uma coluna de número do mês para usar como característica na regressão (Label enconde)
    monthly_data['month_num'] = monthly_data['year_month'].dt.month + 12 * (monthly_data['year_month'].dt.year - monthly_data['year_month'].dt.year.min())

    # Filtrar dados para o produto específico
    product_data = monthly_data[monthly_data['product_id'] == product_id]
    
    # Verificar se há dados suficientes para treinar o modelo
    if len(product_data) < 2:
        return pd.DataFrame({'year_month': [], 'predicted_quantity': []})

    # Selecionar os últimos dois meses
    recent_data = product_data.tail(12)
    
    # Características (X) e variável alvo (y)
    x = product_data[['month_num']]
    y = product_data['quantity']
    
    # Treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(x, y)
    
    # Fazer previsão para o próximo mês
    next_month_num = recent_data['month_num'].max() + 1
    next_month = pd.DataFrame({'month_num': [next_month_num]})
    next_month_prediction = model.predict(next_month)

    # Criar um DataFrame com a previsão para o próximo mês
    next_month_date = pd.date_range(start=recent_data['year_month'].max() + pd.DateOffset(months=1), periods=1, freq='M').strftime('%d-%m-%Y')
    prediction_df = pd.DataFrame({'year_month': next_month_date, 'predicted_quantity': next_month_prediction})

    return {
        'year_month': prediction_df['year_month'][0],
        'predicted_quantity': prediction_df['predicted_quantity'][0]
    }


if __name__ == '__main__':

    project_dir = os.getcwd()
    # csv_file = os.path.join(project_dir, 'assets', 'product_entries.csv')
    csv_file = os.path.join(project_dir, 'assets', 'teste.csv')

    predict = train_and_predict(csv_file=csv_file, product_id='bea127ac-db11-4671-b988-11046e2d2961') 
    logger.success(predict['predicted_quantity'])
    pass