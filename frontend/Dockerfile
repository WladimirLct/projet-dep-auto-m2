FROM python:3.12-slim-bullseye AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project --no-editable

FROM python:3.12-slim-bullseye

# create nonroot user
ARG USERNAME=nonroot
ARG USER_UID=65532
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /app

# Copy application files
COPY main.py /app/main.py

# Copy the environment
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/.venv/bin/python/site-packages"

EXPOSE 8500

USER $USERNAME

# Specify the Python executable and run the Streamlit app
CMD ["/app/.venv/bin/python", "-m", "streamlit", "run", "main.py", "--server.port", "8500"]
