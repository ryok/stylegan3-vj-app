FROM python:3.8

EXPOSE 60001
WORKDIR /app

RUN pip install poetry
COPY pyproject.toml .
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port", "60001"]