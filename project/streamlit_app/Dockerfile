FROM tensorflow/tensorflow:2.9.2-gpu


WORKDIR /app
RUN pip3 install --upgrade pip
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "streamlit", "run",  "app.py"]
