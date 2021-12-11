#!/bin/bash

# Download test dataset
aws s3 cp s3://drivendata-competition-depth-estimation-public/test_videos_downsampled/ /workdir/data/videos/test/ --no-sign-request --recursive

# Download train dataset
aws s3 cp s3://drivendata-competition-depth-estimation-public/train_videos_downsampled/ /workdir/data/videos/train/ --no-sign-request --recursive
