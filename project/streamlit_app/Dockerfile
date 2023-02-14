FROM tensorflow/tensorflow:2.9.2-gpu


WORKDIR /app
RUN apt-get update && apt-get install -y libgl1
RUN apt-get install -y libpangocairo-1.0-0
RUN pip3 install --upgrade pip
COPY . .
WORKDIR /app/streamlit_app
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

CMD [ "streamlit", "run",  "app.py"]
