FROM python:3

COPY ./requirement.txt /app/requirement.txt
COPY ./skin_cancer_model-0.1.0.pkl /app/skin_cancer_model-0.1.0.pkl

WORKDIR /app

RUN pip install -r requirement.txt

COPY . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
