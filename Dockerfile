FROM python:3

RUN pip install --upgrade pip
RUN pip install torch  # For better caching, torch is a 700+ MB download
WORKDIR /app/
ADD ./requirements.txt .
RUN pip install -r ./requirements.txt

ARG CONFIG=./web-proj/config.yaml
ARG CHECKPOINT=./web-proj/latest.cp

RUN mkdir ./workdir/
ADD $CONFIG ./workdir/
ADD $CHECKPOINT ./workdir/

ENV ZGAN_CONFIG=$CONFIG
ENV ZGAN_CHECKPOINT=$CHECKPOINT

ADD dirk/ dirk/
ADD web/ web/
ADD web-proj/ web-proj/

CMD ["gunicorn", \
     "--workers=2", \
     "--bind=0.0.0.0:8080", \
     "web.uwsgi:app"]
