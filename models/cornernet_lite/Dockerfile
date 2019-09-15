FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    wget \
    g++ \
    make \
    git

COPY repo/CornerNet-Lite /CornerNet-Lite
RUN mkdir -p /CornerNet-Lite/data/coco
COPY repo/coco /CornerNet-Lite/data/coco

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh -O /anaconda.sh
RUN bash /anaconda.sh -b -p /anaconda
ENV PATH="/anaconda/bin:${PATH}"

RUN eval "$(/anaconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda activate \
    && conda create --name CornerNet_Lite --file /CornerNet-Lite/conda_packagelist.txt --channel pytorch \
    && conda activate CornerNet_Lite \
    && cd /CornerNet-Lite/core/models/py_utils/_cpools/ \
    && python setup.py install --user \
    && cd /CornerNet-Lite/core/external \
    && make

RUN cd /CornerNet-Lite/data/coco/PythonAPI \
    && make install \
    && mkdir -p /CornerNet-Lite/data/coco/images/trainval2014 \
    && mkdir -p /CornerNet-Lite/data/coco/images/minival2014 \
    && mkdir -p /CornerNet-Lite/data/coco/images/testdev2017 \
    && mkdir -p /CornerNet-Lite/data/coco/annotations

COPY scripts/* /CornerNet-Lite

WORKDIR /CornerNet-Lite
ENTRYPOINT [""]
