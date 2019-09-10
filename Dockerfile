FROM amsterdam/python
MAINTAINER datapunt@amsterdam.nl

ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN adduser --system datapunt
RUN mkdir -p /app && chown datapunt /app
RUN mkdir -p /output && chown datapunt /output && chmod ga+w /output
RUN mkdir -p /cache && chown datapunt /cache
USER datapunt

COPY . /app/
COPY run_* /app/
