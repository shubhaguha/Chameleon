FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Only the docker CLI is needed (it talks to the host daemon through the mounted docker.sock).
# Debian's docker.io client is too old for Docker Engine >= 29 (minimum API 1.44).
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

RUN pip install  --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
