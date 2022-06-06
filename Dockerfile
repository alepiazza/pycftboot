FROM wlandry/sdpb:2.5.1

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -qq update && \
    apt-get -qq install -y python3-pip

COPY . /root/pycftboot

RUN pip install /root/pycftboot
