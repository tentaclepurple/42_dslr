FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

RUN echo 'alias norminette="flake8"' >> /root/.bashrc

CMD ["top", "-b"]