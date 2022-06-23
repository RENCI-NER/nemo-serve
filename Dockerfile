FROM nvcr.io/nvidia/nemo:1.4.0

RUN mkdir /var/web
WORKDIR /var/web

RUN git clone https://github.com/RENCI-NER/nemo-serve.git

WORKDIR /var/web/nemo-serve

RUN pip install -r requirements.txt

ENV PYTHONPATH=/var/web/nemo-serve

ENTRYPOINT uvicorn

CMD ['src.server:app']

