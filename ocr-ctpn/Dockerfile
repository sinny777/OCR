FROM ubuntu16-python35

LABEL maintainer=gurvsin3@in.ibm.com

ENV BUILD_VERSION="V1"

USER root
ENV DEBIAN_FRONTEND noninteractive

# bring system up-to-date
RUN apt-get update -qq && \
    apt-get upgrade -qqy

# Below dependencies required for OpenCV 
RUN apt install -y libsm6 libxext6 libxrender-dev
RUN apt install -y tesseract-ocr
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN apt-get update

# Install Python modules
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
     && rm -rf ~/.cache/pip 

#Change directory so that our commands run inside this new directory
WORKDIR /usr/src/app

COPY . .

RUN chmod 755 entrypoint.sh
RUN chmod 755 build.sh

RUN ./build.sh

ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
CMD ["/bin/bash"]

# EXPOSE 80 8081
