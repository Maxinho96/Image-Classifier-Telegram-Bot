FROM python:3.7

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -U -r requirements.txt

RUN mkdir /app
ADD . /app
WORKDIR /app

CMD python /app/bot.py