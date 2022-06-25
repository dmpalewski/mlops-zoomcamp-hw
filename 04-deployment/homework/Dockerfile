FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

WORKDIR /app

COPY [ "starter.py", "./" ]

CMD ["python", "starter.py", "--year", "2021", "--month", "4"]

