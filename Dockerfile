FROM ubuntu:18.04



WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update
RUN apt-get install -y python3.6-dev \
                       python3-pip \
                       wget \
                       build-essential \
                       software-properties-common \
                       apt-utils \
                       ffmpeg \
                       libsm6 \
                       libxext6


RUN wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
RUN tar -xzf Python-3.8.10.tgz
RUN cd Python-3.8.10/ && ./configure --with-ensurepip=install && make && make install

RUN apt-get update
RUN apt-get install -y libgdal-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r /opt/predict/requirements.txt
RUN ldconfig
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

ENTRYPOINT [ "/usr/bin/python3", "/opt/3d_entropy.py" ]
