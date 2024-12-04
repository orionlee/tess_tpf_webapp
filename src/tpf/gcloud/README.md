# Google Cloud Run Deployment

This directory contains artifacts to deploy the source to Google Cloud Run.

Run [`assemble.sh`](assemble.sh) and follow the instructions to deploy using `gcloud` command.

## Continuous Deployment

The [`cloudbuild.yaml`](cloudbuild.yaml) is primarily meant to be used with [Google Cloud Continuous Deployment](https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build). In theory it can be used in place of `assemble.sh` + `gcloud` above, but it is more cumbersome, and might require permissions tweaks on your Google Cloud account. Example:

```sh
gcloud builds submit --config=src/tpf/gcloud/cloudbuild.yaml  --substitutions=_DEPLOY_REGION="us-west1",_AR_HOSTNAME="us-west1-docker.pkg.dev",_TRIGGER_ID="manual-cli",_SERVICE_NAME="tess-tpf",REPO_NAME="tess_tpf_webapp",COMMIT_SHA="`git rev-parse HEAD`"
```

To set up continuous deployment, create a trigger and

- specify `src/tpf/gcloud/cloudbuild.yaml` as the Cloud Build configuration file
- Add the following ignore file filters (to avoid unnecessary builds):

```txt
README.md
src/tpf/gcloud/README.md
*.ipynb
assets/*
```
