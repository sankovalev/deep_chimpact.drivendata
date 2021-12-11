#!/bin/bash

echo "Use 7 models to make level1-predictions (9106 images). Each model has 5 folds, so make 35 predictions in total with TTA"
python scripts/predict.py --models /workdir/models/saved_models/n09_e00-fold0/model-077-1.035534.pth /workdir/models/saved_models/n09_e00-fold1/model-098-0.953209.pth /workdir/models/saved_models/n09_e00-fold2/model-019-0.985004.pth /workdir/models/saved_models/n09_e00-fold3/model-033-1.016116.pth /workdir/models/saved_models/n09_e00-fold4/model-033-0.884321.pth --name n09_e00 -tta &
python scripts/predict.py --models /workdir/models/saved_models/n09_e01-fold0/model-029-1.094792.pth /workdir/models/saved_models/n09_e01-fold1/model-057-0.971676.pth /workdir/models/saved_models/n09_e01-fold2/model-073-0.973727.pth /workdir/models/saved_models/n09_e01-fold3/model-065-1.039491.pth /workdir/models/saved_models/n09_e01-fold4/model-080-0.891617.pth --name n09_e01 -tta &
python scripts/predict.py --models /workdir/models/saved_models/n09_e02-fold0/model-074-1.056197.pth /workdir/models/saved_models/n09_e02-fold1/model-063-0.975252.pth /workdir/models/saved_models/n09_e02-fold2/model-021-0.999933.pth /workdir/models/saved_models/n09_e02-fold3/model-027-1.069523.pth /workdir/models/saved_models/n09_e02-fold4/model-054-0.892059.pth --name n09_e02 -tta &
wait
python scripts/predict.py --models /workdir/models/saved_models/n09_e03-fold0/model-033-1.038391.pth /workdir/models/saved_models/n09_e03-fold1/model-038-0.988344.pth /workdir/models/saved_models/n09_e03-fold2/model-007-1.005588.pth /workdir/models/saved_models/n09_e03-fold3/model-023-1.073944.pth /workdir/models/saved_models/n09_e03-fold4/model-044-0.909110.pth --name n09_e03 -tta &
python scripts/predict.py --models /workdir/models/saved_models/n09_e04-fold0/model-058-1.008950.pth /workdir/models/saved_models/n09_e04-fold1/model-048-0.964409.pth /workdir/models/saved_models/n09_e04-fold2/model-049-1.026899.pth /workdir/models/saved_models/n09_e04-fold3/model-049-1.036091.pth /workdir/models/saved_models/n09_e04-fold4/model-074-0.917152.pth --name n09_e04 -tta &
wait
python scripts/predict.py --models /workdir/models/saved_models/n09_e05-fold0/model-043-1.044745.pth /workdir/models/saved_models/n09_e05-fold1/model-059-0.992952.pth /workdir/models/saved_models/n09_e05-fold2/model-041-0.987204.pth /workdir/models/saved_models/n09_e05-fold3/model-019-1.059817.pth /workdir/models/saved_models/n09_e05-fold4/model-078-0.896042.pth --name n09_e05 -tta &
python scripts/predict.py --models /workdir/models/saved_models/n09_e08-fold0/model-110-1.059046.pth /workdir/models/saved_models/n09_e08-fold1/model-038-0.947302.pth /workdir/models/saved_models/n09_e08-fold2/model-069-1.011137.pth /workdir/models/saved_models/n09_e08-fold3/model-031-1.047628.pth /workdir/models/saved_models/n09_e08-fold4/model-067-0.876613.pth --name n09_e08 -tta &
wait

echo "Create level2 dataset. Predict each fold on correspond valid fold"
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e00-fold0/model-077-1.035534.pth --fold 0 --name n09_e00 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e00-fold1/model-098-0.953209.pth --fold 1 --name n09_e00 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e00-fold2/model-019-0.985004.pth --fold 2 --name n09_e00 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e00-fold3/model-033-1.016116.pth --fold 3 --name n09_e00 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e00-fold4/model-033-0.884321.pth --fold 4 --name n09_e00 -tta &
wait

python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e01-fold0/model-029-1.094792.pth --fold 0 --name n09_e01 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e01-fold1/model-057-0.971676.pth --fold 1 --name n09_e01 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e01-fold2/model-073-0.973727.pth --fold 2 --name n09_e01 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e01-fold3/model-065-1.039491.pth --fold 3 --name n09_e01 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e01-fold4/model-080-0.891617.pth --fold 4 --name n09_e01 -tta &
wait

python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e02-fold0/model-074-1.056197.pth --fold 0 --name n09_e02 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e02-fold1/model-063-0.975252.pth --fold 1 --name n09_e02 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e02-fold2/model-021-0.999933.pth --fold 2 --name n09_e02 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e02-fold3/model-027-1.069523.pth --fold 3 --name n09_e02 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e02-fold4/model-054-0.892059.pth --fold 4 --name n09_e02 -tta &
wait

python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e03-fold0/model-033-1.038391.pth --fold 0 --name n09_e03 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e03-fold1/model-038-0.988344.pth --fold 1 --name n09_e03 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e03-fold2/model-007-1.005588.pth --fold 2 --name n09_e03 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e03-fold3/model-023-1.073944.pth --fold 3 --name n09_e03 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e03-fold4/model-044-0.909110.pth --fold 4 --name n09_e03 -tta &
wait

python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e04-fold0/model-058-1.008950.pth --fold 0 --name n09_e04 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e04-fold1/model-048-0.964409.pth --fold 1 --name n09_e04 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e04-fold2/model-049-1.026899.pth --fold 2 --name n09_e04 -tta &
wait
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e04-fold3/model-049-1.036091.pth --fold 3 --name n09_e04 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e04-fold4/model-074-0.917152.pth --fold 4 --name n09_e04 -tta &
wait

python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e05-fold0/model-043-1.044745.pth --fold 0 --name n09_e05 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e05-fold1/model-059-0.992952.pth --fold 1 --name n09_e05 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e05-fold2/model-041-0.987204.pth --fold 2 --name n09_e05 -tta &
wait
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e05-fold3/model-019-1.059817.pth --fold 3 --name n09_e05 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e05-fold4/model-078-0.896042.pth --fold 4 --name n09_e05 -tta &
wait

python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e08-fold0/model-110-1.059046.pth --fold 0 --name n09_e08 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e08-fold1/model-038-0.947302.pth --fold 1 --name n09_e08 -tta &
wait
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e08-fold2/model-069-1.011137.pth --fold 2 --name n09_e08 -tta &
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e08-fold3/model-031-1.047628.pth --fold 3 --name n09_e08 -tta &
wait
python scripts/predict_on_fold.py --models /workdir/models/saved_models/n09_e08-fold4/model-067-0.876613.pth --fold 4 --name n09_e08 -tta

echo "Stack folds together"
python scripts/stack_folds.py --folds /workdir/data/level2/n09_e00_fold0.csv /workdir/data/level2/n09_e00_fold1.csv /workdir/data/level2/n09_e00_fold2.csv /workdir/data/level2/n09_e00_fold3.csv /workdir/data/level2/n09_e00_fold4.csv --name n09_e00 &
python scripts/stack_folds.py --folds /workdir/data/level2/n09_e01_fold0.csv /workdir/data/level2/n09_e01_fold1.csv /workdir/data/level2/n09_e01_fold2.csv /workdir/data/level2/n09_e01_fold3.csv /workdir/data/level2/n09_e01_fold4.csv --name n09_e01 &
python scripts/stack_folds.py --folds /workdir/data/level2/n09_e02_fold0.csv /workdir/data/level2/n09_e02_fold1.csv /workdir/data/level2/n09_e02_fold2.csv /workdir/data/level2/n09_e02_fold3.csv /workdir/data/level2/n09_e02_fold4.csv --name n09_e02 &
python scripts/stack_folds.py --folds /workdir/data/level2/n09_e03_fold0.csv /workdir/data/level2/n09_e03_fold1.csv /workdir/data/level2/n09_e03_fold2.csv /workdir/data/level2/n09_e03_fold3.csv /workdir/data/level2/n09_e03_fold4.csv --name n09_e03 &
python scripts/stack_folds.py --folds /workdir/data/level2/n09_e04_fold0.csv /workdir/data/level2/n09_e04_fold1.csv /workdir/data/level2/n09_e04_fold2.csv /workdir/data/level2/n09_e04_fold3.csv /workdir/data/level2/n09_e04_fold4.csv --name n09_e04 &
python scripts/stack_folds.py --folds /workdir/data/level2/n09_e05_fold0.csv /workdir/data/level2/n09_e05_fold1.csv /workdir/data/level2/n09_e05_fold2.csv /workdir/data/level2/n09_e05_fold3.csv /workdir/data/level2/n09_e05_fold4.csv --name n09_e05 &
python scripts/stack_folds.py --folds /workdir/data/level2/n09_e08_fold0.csv /workdir/data/level2/n09_e08_fold1.csv /workdir/data/level2/n09_e08_fold2.csv /workdir/data/level2/n09_e08_fold3.csv /workdir/data/level2/n09_e08_fold4.csv --name n09_e08 &
wait

echo "Add labels and create full level2 dataset"
python scripts/create_level2_dataset.py --meta_train /workdir/data/meta/train_metadata_v3.csv --meta_test /workdir/data/meta/test_metadata_v4.csv \
                                        --train_files /workdir/data/level2/n09_e00.csv /workdir/data/level2/n09_e01.csv /workdir/data/level2/n09_e02.csv /workdir/data/level2/n09_e03.csv /workdir/data/level2/n09_e04.csv /workdir/data/level2/n09_e05.csv /workdir/data/level2/n09_e08.csv \
                                        --test_files /workdir/data/predictions/raw_n09_e00.csv /workdir/data/predictions/raw_n09_e01.csv /workdir/data/predictions/raw_n09_e02.csv /workdir/data/predictions/raw_n09_e03.csv /workdir/data/predictions/raw_n09_e04.csv /workdir/data/predictions/raw_n09_e05.csv /workdir/data/predictions/raw_n09_e08.csv \
                                        --name final

echo "Fit & predict"
python scripts/fit_predict_level2.py --train_file /workdir/data/level2/train_final.csv --test_file /workdir/data/level2/test_final.csv --name final

echo "Apply heuristics, postprocessing and convert to valid format"
python scripts/make_submit.py --name final

echo "Done! Submission file was successfully saved to /workdir/data/predictions/final.csv"
date +%m.%d.%Y-%H:%M:%S
