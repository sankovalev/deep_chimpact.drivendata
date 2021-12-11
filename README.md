# DEEP CHIMPACT CHALLENGE

## Requirements
- Ubuntu >= 18.04
- Docker version >= 19.03.3
- Any GPU with at least 8 Gb of memory
- CUDA Version: 11.4 (Driver Version: 470)
---
## Steps to reproduce final submission:
0. Download `models` from [google drive](https://drive.google.com/drive/folders/1YoibyArJoocgjbXE7soBL5dqEjyPYKUn?usp=sharing) and put in the root of folder.
1. Build docker image, run container and go inside.
```bash
make build run-dev exec
```
2. Download **downsampled** videos from S3 or *skip this step* if all videos are already in the `data/` folder.
```bash
sh scripts/download_data.sh
```
The final structure of `/workdir/` should look like this:
```
|-- data/
|  |-- meta/
|  |-- predictions/
|  |-- level2/
|  |-- videos/
|  |  |-- train/
|  |  |-- test/
|  |-- submission_format.csv
|-- models/
|  |-- pretrained/
|  |-- saved_models/
|-- scripts/
|-- src/
...
```
3. Prepare data: unpack videos & run depth-estimation pretrained net.
```bash
sh scripts/prepare_data.sh
```
4. Make two-stage predictions.
*Note: this step takes about 3.5 hours.*
```
sh scripts/predict.sh
```
---
## Brief description of solution
- First, I extracted the frames specified in the meta information from all the videos. I didn't use the video anywhere else, then I worked only with frames.
- After analyzing the meta-information, I found that a significant part of the frames either lack information about the bounding boxes with animals, or they are incorrect.
In the description of the competition it was said that *"Since bounding boxes are model generated they are not always correct. They are provided only as a helpful reference, not as a ground truth."*
Therefore, I decided to improve this aspect and enrich the meta-information.
- I selected only those examples from both the train and the test for which confidence was higher than 0.5.
From these frames and annotations to them, I made new dataset for object detection task and trained [YOLOv5](https://github.com/ultralytics/yolov5).
- On the frames that did not get into the new dataset, I made predictions with a trained detector.
After I threw out the false and incorrect ones, updated the files with meta information, and also formed a blacklist of frames on which animals could not be detected. This is how files with "_vN" suffixes appeared in `data/meta`.
**I repeated this iteration several times**, gradually increasing the frame size for the detector. This made it possible to find animals in those frames where they are far from the camera trap and have small sizes.
- Then I applied a pre-trained large depth-estimation [DPT](https://github.com/isl-org/DPT) model to each frame from train & test.
- Finally, I trained an ensemble of efficientnets by cutting a crop with an animal from the original image and from the corresponding depth map. The target was the distance to the animal from the train_labels.csv file. The size of the input images is 224x224, TTA with gmean averaging was used for predictions.
- To improve the final score, I trained a second-level linear model. The final submission simply uses the weighted sum of the ensemble's predictions.

##### To reproduce each described pipeline step, see readme from `scripts/` folder.
