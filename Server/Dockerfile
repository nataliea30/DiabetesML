# syntax=docker/dockerfile:1

FROM python:3.11-slim-buster

WORKDIR /Server

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "/Server/app.py"]
EXPOSE 5003