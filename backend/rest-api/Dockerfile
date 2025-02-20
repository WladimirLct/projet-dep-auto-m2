FROM python:3.12-slim-bullseye

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# create nonroot user
ARG USERNAME=nonroot
ARG USER_UID=65532
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project --no-editable

COPY main.py /app/main.py
COPY config/ /app/config
COPY routes/ /app/routes

RUN rm -rf /usr/local/bin/uv

EXPOSE 9000

USER $USERNAME

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]