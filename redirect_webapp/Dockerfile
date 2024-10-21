# Create an image based on distroless python3 debian12 / bookworm
# - it bundles with pyhton 3.11,
#   so the rest of the build needs to follow it

# Derived from:
# https://github.com/GoogleContainerTools/distroless/blob/main/examples/python3-requirements/Dockerfile
# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/run/helloworld/Dockerfile
#

# 1. Build a virtualenv to get production dependencies.
#    Use the official lightweight Python image.
#    https://hub.docker.com/_/python
FROM python:3.11-slim-bookworm AS build-venv
COPY requirements.txt /requirements.txt
RUN python3 -m venv /venv
RUN /venv/bin/pip install --disable-pip-version-check --no-cache-dir -r /requirements.txt


# 2. Copy the virtualenv into a distroless image
FROM gcr.io/distroless/python3-debian12
# copy dependencies and set the python search path
COPY --from=build-venv /venv /venv
ENV PYTHONPATH /venv/lib/python3.11/site-packages
# copy the sources
COPY . /app
WORKDIR /app

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# # debug with :debug image
#SHELL ["/busybox/sh", "-c"]
#RUN /usr/bin/python3 --version
#RUN ls /venv/lib/python3.11/site-packages
#RUN /usr/bin/python3  -c "import gunicorn; print(gunicorn.__version__)"

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
#
#  - Need explicit python: the script /venv/bin/gunicorn does not work by itself in the setup
#  - Cannot get $PORT passed to ENTRYPOINT in vector form: rely on gunicorn to pick up $PORT (its default behavior).
ENTRYPOINT ["/usr/bin/python3", "/venv/bin/gunicorn",  "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
# the non-vector form does not work with shellless distroless
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
