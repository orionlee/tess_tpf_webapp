# Based on the samples from:
# https://github.com/GoogleContainerTools/distroless/blob/main/examples/python3-requirements/Dockerfile
# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/run/helloworld/Dockerfile

# Build a virtualenv using the appropriate Debian release
# * Install gcc and libc6-dev to compile C Python modules
# * In the virtualenv: Update pip setuputils and wheel to support building new packages
FROM python:3.13-slim-trixie AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes gcc libc6-dev && \
    # Symlink the distroless path for python: /usr/bin/python to the build
    # image path: /usr/local/bin/python to ensure the runtime image's venv has
    # the right python paths.
    ln -s /usr/local/bin/python /usr/bin/python && \
    /usr/bin/python -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Build the virtualenv as a separate step: Only re-execute this step when requirements.txt changes
FROM build AS build-venv
COPY requirements.txt /requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r /requirements.txt


#
# Create the runtime distroless-based image
#

# Copy the virtualenv into a distroless image
FROM gcr.io/distroless/python3-debian13
COPY --from=build-venv /venv /venv
COPY . /app
WORKDIR /app

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Note: In gcloud service dashboard,
#       add environment variable BOKEH_ALLOW_WS_ORIGIN
#       and set it to the public hostname of the deployed service.
# See: https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html#network-configuration
#
# MUST use vector form as distroless has no shell
# OPEN: env vars not supported without shell. Limitations for now:
# - hardcoded the port to 8080
# - extra args via $BOKEH_EXTRA_ARGS not supported (e.g., to tweak bokeh logging)
ENTRYPOINT ["/venv/bin/python3", "/venv/bin/bokeh", "serve", "--port", "8080", "tpf"]
# non-vector form (use shell): CMD exec bokeh serve --port $PORT tpf $BOKEH_EXTRA_ARGS
