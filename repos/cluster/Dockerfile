FROM python:3.7

ADD . /cluster
WORKDIR cluster
# RUN pip3 freeze > requirements.txt
RUN pip3 install pipreqs
RUN pipreqs .
RUN pip3 install -r requirements.txt
RUN pip3 install django-bootstrap4
RUN pip3 install google-cloud-storage
RUN python3 manage.py migrate

EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]


