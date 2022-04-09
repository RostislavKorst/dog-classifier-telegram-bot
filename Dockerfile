FROM python:3.8-slim-buster

WORKDIR /app

# cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py app.py
COPY model ./model/

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]