FROM apache/airflow:2.7.1-python3.10

COPY requirements.txt /v_requirements.txt
RUN pip install --no-cache-dir -r /v_requirements.txt
