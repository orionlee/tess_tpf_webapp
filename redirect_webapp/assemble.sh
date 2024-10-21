#!/bin/sh

# Assemble TESS TCEs webapp that can be source-deployed to Google Cloud Run
#

base=`dirname $0`
dest=$1

set -e

mkdir -p $dest

cp --update --archive  $base/*  $dest
cp --update --archive  $base/.*  $dest

echo Sources assembled:
ls -l $dest

echo
echo Sources assembled. You can do the following for actual deployment:
echo
echo cd $dest
echo "# sanity test locally"
echo python main.py
echo "# actual deployment with Google Cloud SDK"
echo gcloud run deploy tess-skyview --source .
