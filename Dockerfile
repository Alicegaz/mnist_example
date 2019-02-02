FROM python:alpine3.7
FROM continuumio/miniconda3
RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

COPY . /app
COPY ./app/requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt
WORKDIR /app
