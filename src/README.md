# Google Cloud Continuous Deployment

This branch is used for [Google Cloud Continuous Deployment](https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build).


gcloud deployment related artifacts, in [tpf/gcloud](tpf/gcloud) directory, are copied here so that the files layout can be picked up by Google Cloud Continuous Deployment.

Note: if the deployment artifacts in `tpf/gcloud` are modified, the version here need to be updated correspondingly.


## Caveats

- Specifying using `Dockerfile` (at `src/`) to define build Configuration: the console UI did not appear to set it correctly. It needs to be re-specified by editing the created build trigger.
- Identity and Access Management (IAM) API needs to be enabled.
