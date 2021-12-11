### Start training:
1. Single GPU
```bash
python3 train.py
```
2. Multiple GPU (also set WORLD_SIZE environment variable)
```bash
python3 -m torch.distributed.launch --nproc_per_node=1 train.py
```

#### Note: for additional options for all scripts, see `--help`


## Full pipeline:
0. Run `unpack_videos.py`
1. Run `run_dpt.py`
2. Run `visualize_datasets.py`
3. Run `visualize_ground_truth.py`
4. Run `create_yolov5_dataset.py`
5. Train yolov5 (see below)
6. Predict with yolov5 (see below)
7. Manually update meta and black list
8. Run `create_folds.py`
9. Fill in config.py and run `train.py`
10. Run `predict.py`
11. Run `make_submit.py`
12. Run `combine_submits.py`
13. Run `visualize_predictions.py`

## Level2 pipeline:
14. Run `predict_on_fold.py`
15. Run `stack_folds.py`
16. Run `create_level2_dataset.py`
17. Run `fit_predict_level2.py`


#### How to train, validate and predict with YOLOv5.
Create valid manually. Move all files started with `x`, `y` and `z` to valid folder.
```bash
python src/yolov5/train.py --img 640 --batch 12 --epochs 40 --data deep_chimpact.yaml --weights models/pretrained/yolov5x.pt
python src/yolov5/detect.py --weights=src/yolov5/runs/train/exp/weights/best.pt --source=src/yolov5/datasets/deep_chimpact/images/test/ --conf-thres=0.15 --max-det=1 --imgsz=640 --save-txt --save-conf --name=exp
```
