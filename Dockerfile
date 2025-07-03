# Make python base image
FROM python:3.11-slim AS build

# install curl
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
	curl \
	build-essential \
    libffi-dev \
	; \
	rm -rf /var/lib/apt/lists/*

# install poetry
ENV POETRY_HOME /opt/poetry
ENV POETRY_VERSION "2.0.1"
RUN curl -sSL https://install.python-poetry.org | python3 -

# add poetry to path
ENV PATH /opt/poetry/bin:${PATH}

# set working directory
WORKDIR /app

# create venv
ENV POETRY_VIRTUALENVS_IN_PROJECT=1

# copy over dependency files
COPY pyproject.toml poetry.lock ./

# install dependencies
RUN poetry install --only main --no-root

FROM python:3.11-slim AS run
WORKDIR /app
COPY --from=build /app/.venv /app/.venv

# update path for venv
ENV PATH /app/.venv/bin:${PATH}

# copy over our src
COPY . . 

HEALTHCHECK --start-period=30s CMD python -c "import requests; requests.get('http://localhost:8000/app/health', timeout=2)"

# run uvicorn
CMD uvicorn osha_app.src.main:app --host 0.0.0.0