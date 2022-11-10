FROM python:3.9.12-slim-buster

COPY requirements.txt .

RUN apt update && apt install -fy rustc

RUN pip install -r requirements.txt

COPY app.py .
COPY models models

RUN mkdir /.cache && chown -R 1100:1100 /.cache ;

CMD waitress-serve --port=5001 app:app
