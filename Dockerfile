FROM ubuntu:latest
LABEL authors="naresh"

FROM python:3.10.14-slim
WORKDIR /src
COPY . /src
RUN pip install flask gunicorn tensorflow pandas numpy pickle-mixin
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]


ENTRYPOINT ["top", "-b"]