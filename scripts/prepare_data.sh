#!/bin/bash

# Extract target frames from test videos
python scripts/unpack_videos.py --videos=/workdir/data/videos/test/ --meta=/workdir/data/meta/test_metadata_v4.csv --output=data/images/test/

# Extract target frames from train videos
python scripts/unpack_videos.py --videos=/workdir/data/videos/train/ --meta=/workdir/data/meta/train_metadata_v3.csv --output=data/images/train/

# Predict depth on test frames
python scripts/run_dpt.py --input_path=/workdir/data/images/train/ --output_path=/workdir/data/images/train_dpt_large/ --model_type=dpt_large

# Predict depth on train frames
python scripts/run_dpt.py --input_path=/workdir/data/images/test/ --output_path=/workdir/data/images/test_dpt_large/ --model_type=dpt_large
