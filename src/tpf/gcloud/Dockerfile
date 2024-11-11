# Based on the sample from:
# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/run/helloworld/Dockerfile

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install git (needed for the installing lightkurve from github source directly)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Note: In gcloud service dashboard,
#       add environment variable BOKEH_ALLOW_WS_ORIGIN
#       and set it to the public hostname of the deployed service.
# See: https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html#network-configuration
CMD exec bokeh serve --port $PORT tpf
