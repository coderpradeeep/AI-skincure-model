FROM python:3

COPY ./requirements.txt /app/requirements.txt
COPY ./skin_cancer_model-0.1.0.pkl /app/skin_cancer_model-0.1.0.pkl

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
