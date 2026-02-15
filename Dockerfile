# Use a slim Python 3.13 image
FROM python:3.13-slim

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --frozen ensures we use the exact versions from uv.lock
# --no-dev excludes development dependencies
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit application
# We use 'uv run' to ensure the environment is correctly loaded
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
