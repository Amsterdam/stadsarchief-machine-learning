FROM amsterdam/python
MAINTAINER datapunt@amsterdam.nl

ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY requirements/* /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN adduser --system datapunt
RUN mkdir -p /app && chown datapunt /app && chmod g+s /app
RUN mkdir -p /output && chown datapunt /output && chmod ga+w /output
RUN mkdir -p /cache && chown datapunt /cache

COPY src /app/src
COPY scripts /app/scripts
COPY run_* /app/

USER datapunt
