FROM python:3.11.5
EXPOSE 8010

COPY . . 

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /src

#RUN pip install -r requirements.txt

CMD [ "python", "app.py" ]