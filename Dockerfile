# syntax=docker/dockerfile:1.2
FROM python:latest

# Install linux apps
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         g++ \
         gcc musl-dev \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/pkg/challenge

# Set ENV variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/pkg/challenge:${PATH}"

# Add project files and permission
ADD challenge /opt/ml/pkg/challenge
# Include entrypoint script(s)
RUN chmod +x /opt/ml/pkg/challenge/train
RUN chmod +x /opt/ml/pkg/challenge/serve

# Build the model
RUN make build
