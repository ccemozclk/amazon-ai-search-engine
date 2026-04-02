
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

WORKDIR /app

COPY requirements/etl.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r etl.txt

COPY config ./config/
COPY src ./src/

CMD ["python", "-m", "src.pipelines.data_ingestion"]
