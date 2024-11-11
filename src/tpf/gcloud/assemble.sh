#!/bin/sh

proj_root=../../../

base=`dirname $0`
dest=$1

if [ "$dest" == "" ]; then
    dest=$base/../../../build/tess-tpf
fi

set -e

mkdir -p $dest
mkdir -p $dest/tpf

cp --update --archive  $base/../*.py  $dest/tpf
cp --update --archive  $base/*  $dest
cp --update --archive  $base/.*  $dest

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
echo Additional envrionment variables reocommended:
echo 1. Use CDN to serve bokeh assets: javascripts, etc.
echo BOKEH_RESOURCES=cdn
echo 2. Define the Python log level of the webapp, e.g.,
echo TESS_TPF_WEBAPP_LOGLEVEL=INFO
echo
