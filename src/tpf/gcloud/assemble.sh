#!/usr/bin/env bash

base=`dirname $0`
dest=$base/../../../build/tess-tpf

set -e

# the commit SHA
# - in gcloud continuous deployment env, it will be supplied as a parameter
#   (git not available in gcloud's bash build step)
# - in local build, we generates it locally
commit_sha=$1
if [ "$commit_sha" == "" ]; then
  commit_sha=`git rev-parse HEAD`
  echo To save commit SHA in build: $commit_sha
fi

# first clean the destination dir to ensure a clean build
if [ -d "$dest" ]; then
  rm -fr $dest
fi

mkdir -p $dest
mkdir -p $dest/tpf

cp --update --archive  $base/../*.py  $dest/tpf
cp --update --archive  --recursive $base/../lk_patch  $dest/tpf
cp --update --archive  $base/*  $dest
cp --update --archive  $base/.*  $dest
# the cloudbuild.yaml is setup for use in continuous deployment
# it MUST not be in $dest ,
# as it would be picked up by "gcloud run deploy --source .", and would create bad builds.
rm -f $dest/cloudbuild.yaml

# save commit SHA to be displayed in the UI.
echo $commit_sha > $dest/tpf/build.txt

ls -l $dest/ $dest/tpf

echo
echo Sources assembled. You can do the following for actual deployment:
echo
echo cd $dest
echo "# sanity test locally"
echo bokeh serve --show tpf
echo "# actual deployment with Google Cloud SDK"
echo gcloud run deploy --source .
echo
echo Note:   The first deployment, the app  would not work, returning blank page
echo         with errors in developer console suggesting failure in opening websockets.
echo Action: In gcloud service dashboard, add environment variable BOKEH_ALLOW_WS_ORIGIN,
echo         set it to the public hostname of the deployed service, and deploy again.
echo
echo Additional environment variables recommended:
echo 1. Use CDN to serve bokeh assets: javascripts, etc.
echo BOKEH_RESOURCES=cdn
echo 2. Define the Python log level of the webapp, e.g.,
echo TESS_TPF_WEBAPP_LOGLEVEL=INFO
echo
