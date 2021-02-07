FROM jupyter/scipy-notebook

USER root

RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt ./
RUN pip install -r requirements.txt  && rm requirements.txt

ENV WORK_DIR ${HOME}/work