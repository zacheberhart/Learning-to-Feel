FROM python:3.7
EXPOSE 8501

RUN apt-get update
RUN apt-get install -y libsndfile1-dev
RUN apt-get install -y ffmpeg

WORKDIR /usr/src/app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .

CMD [ "streamlit", "run", "src/app.py" ]