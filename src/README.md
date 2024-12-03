# Google Cloud Continuous Deployment

This branch is used for [Google Cloud Continuous Deployment](https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build).


gcloud deployment related artifacts, in [tpf/gcloud](tpf/gcloud) directory, are copied here so that the files layout can be picked up by Google Cloud Continuous Deployment.

Note: if the deployment artifacts in `tpf/gcloud` are modified, the version here need to be updated correspondingly.

TODO: create a custom `cloudbuild.yaml` to avoid copying / syncing the artifacts from `tpf/gcloud`. The yaml can be extened from the
inline yaml to first copying the artifacts using bash script. See the [doc](https://cloud.google.com/build/docs/configuring-builds/run-bash-scripts?#running_bash_scripts_on_disk) on cloudbuild.yaml .


## Tips / Caveats

- Setting build configuration to `Dockerfile` would not work. It would build but not deploy the image.
- Instead, use the default "Cloud Build configuration file (yaml or json)", with an inline yaml.
  - The inline yaml would need to be modified. It assumes the build root dir is `.`. But it is `src` in our case.
  - Change the `docker build` arguments of `. -f Dockerfile` to `src -f src/Dockerfile`.

- Add ignore filters:

```
README.md
src/README.md
src/tpf/gcloud/*
*.ipynb
assets/*
```

- Identity and Access Management (IAM) API needs to be enabled.

---

The modified inline build yaml file (substitute the `[the-trigger-id]`):

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - '--no-cache'
      - '-t'
      - >-
        $_AR_HOSTNAME/$PROJECT_ID/cloud-run-source-deploy/$REPO_NAME/$_SERVICE_NAME:$COMMIT_SHA
      - src
      - '-f'
      - src/Dockerfile
    id: Build
  - name: gcr.io/cloud-builders/docker
    args:
      - push
      - >-
        $_AR_HOSTNAME/$PROJECT_ID/cloud-run-source-deploy/$REPO_NAME/$_SERVICE_NAME:$COMMIT_SHA
    id: Push
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk:slim'
    args:
      - run
      - services
      - update
      - $_SERVICE_NAME
      - '--platform=managed'
      - >-
        --image=$_AR_HOSTNAME/$PROJECT_ID/cloud-run-source-deploy/$REPO_NAME/$_SERVICE_NAME:$COMMIT_SHA
      - >-
        --labels=managed-by=gcp-cloud-build-deploy-cloud-run,commit-sha=$COMMIT_SHA,gcb-build-id=$BUILD_ID,gcb-trigger-id=$_TRIGGER_ID
      - '--region=$_DEPLOY_REGION'
      - '--quiet'
    id: Deploy
    entrypoint: gcloud
images:
  - >-
    $_AR_HOSTNAME/$PROJECT_ID/cloud-run-source-deploy/$REPO_NAME/$_SERVICE_NAME:$COMMIT_SHA
options:
  substitutionOption: ALLOW_LOOSE
  logging: CLOUD_LOGGING_ONLY
substitutions:
  _DEPLOY_REGION: us-west1
  _AR_HOSTNAME: us-west1-docker.pkg.dev
  _PLATFORM: managed
  _TRIGGER_ID: [the-trigger-id]
  _SERVICE_NAME: tess-tpf
tags:
  - gcp-cloud-build-deploy-cloud-run
  - gcp-cloud-build-deploy-cloud-run-managed
  - tess-tpf
```
