FROM amsterdam/python
MAINTAINER datapunt@amsterdam.nl

ENV PYTHONUNBUFFERED 1

# TODO use or remove
RUN apt-get update \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#RUN adduser --system datapunt
#RUN mkdir -p /app && chown datapunt /app

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
#USER datapunt

COPY . /app/


