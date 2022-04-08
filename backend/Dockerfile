FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /
RUN git clone https://github.com/NVlabs/stylegan3.git

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    git \
    build-essential \
    libgl1-mesa-dev \
    libsndfile1 \
    ffmpeg

RUN apt-get update
RUN pip install uvicorn
RUN pip install opencv-python
COPY ./pyproject.toml /app/
RUN pip install poetry \ 
    && poetry config virtualenvs.create false \
    && poetry install

# COPY . /app/

EXPOSE 60080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "60080" , "--reload"]