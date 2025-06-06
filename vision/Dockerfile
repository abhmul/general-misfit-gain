FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install -y python-pip

COPY docker_requirements.txt /opt/app/requirements.txt
RUN pip uninstall -y setuptools && pip install setuptools && \
    pip install --upgrade pip && \ 
    pip install -r /opt/app/requirements.txt