FROM python:3.11-slim

# poppler-utils: necesario para pdf2image (pdftoppm)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

EXPOSE 8000
ENV DONUT_MODEL_ID="naver-clova-ix/donut-base"
ENV DONUT_MAX_LENGTH="1024"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]