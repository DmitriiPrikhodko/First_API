FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./main.py /code/main.py

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py", "--server.port=8503"]