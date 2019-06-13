FROM amsterdam/python
MAINTAINER datapunt@amsterdam.nl

ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN adduser --system datapunt
RUN mkdir -p /app && chown datapunt /app
USER datapunt

COPY . /app/


