#!/usr/bin/env bash

echo "RUNNING OCR USING CTPN...."

function install_python {
    wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
    tar xzf Python-3.5.6.tgz
    cd Python-3.5.6
    ./configure --enable-optimizations
    make altinstall

    python3.5 -V

    cd ..

    # add-apt-repository universe
    # apt-get update

    # apt-get install python-pip
    # apt-get install python3-pip

    pip install --no-cache-dir -r /requirements.txt \
    && rm -rf ~/.cache/pip

    python3.5 -c 'import tensorflow as tf; print(tf.__version__)'
}

function install_dependencies1 {
    python3 -V

    # pip install --no-cache-dir -r requirements.txt \
    # && rm -rf ~/.cache/pip

    python3 -c 'import tensorflow as tf; print(tf.__version__)'

}

function install_dependencies2 {
    # wget https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view?usp=sharing
    cd utils/bbox
    chmod +x make.sh
    ./make.sh
    cd ..
    cd ..
}

install_dependencies1
install_dependencies2



