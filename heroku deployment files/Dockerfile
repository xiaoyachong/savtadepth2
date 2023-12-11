FROM python:3.9.10

EXPOSE 7860

WORKDIR /app

COPY requirements_heroku.txt requirements_heroku.txt

RUN pip install -r requirements_heroku.txt

COPY /app /app  
COPY /examples /examples

COPY setup.sh setup.sh

#ENV GRADIO_SERVER_NAME=0.0.0.0 
#ENV GRADIO_SERVER_PORT="$PORT"

CMD ["python", "/app/app_savta.py"]