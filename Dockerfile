FROM apache/airflow:2.9.3-python3.12

WORKDIR /opt/airflow

# Copia e instala dependências (mantém cache entre builds)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt
RUN airflow db init

