FROM tensorflow/tensorflow:1.5.0-gpu-py3

# Defines image with old Tensorflow for CPUs without SSE4.1 supported.
# Bumps python to 3.6

# Bumping python to version 3.6
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3

RUN python3 --version

# Install requirements
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt

COPY desktop-gpu/jupyter_notebook_config.py /root/.jupyter/
