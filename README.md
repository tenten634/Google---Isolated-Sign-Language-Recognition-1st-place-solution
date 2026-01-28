# Google: Isolated-Sign-Language-Recognition-1st-place-solution

see: https://www.kaggle.com/competitions/asl-signs/discussion/406684

## Run as Python
This repo includes a Python version of the notebook in `train.py`.

Example:
```
python train.py \
  --train_filenames "/path/to/tfrecords/*.tfrecords" \
  --train_csv "/path/to/train.csv" \
  --wandb_project isl-1
```

If you place datasets under `datammount/`, defaults are:
```
python train.py \
  --data_dir "./datammount" \
  --tfrecords_subdir "ISLR-5fold" \
  --competition_subdir "Google-Isolated-Sign-Language-Recognition"
```

## Dataset notes (Kaggle)
The notebook uses Kaggle GCS paths (see `GCS_PATH` in `ISLR_1st_place_Hoyeol_Sohn.ipynb`).
Those paths expire, so on Kaggle you need to refresh them via:
`KaggleDatasets.get_gcs_path(dataset_name)`.

For local runs, download the competition data via Kaggle API:
```
kaggle competitions download -c asl-signs -p data/islr
unzip data/islr/asl-signs.zip -d data/islr
```

`train.py` expects TFRecords and `train.csv`. If you do not already have the TFRecords used
in the notebook, you will need to create them from the competition data or obtain the
author's TFRecord dataset and point `--train_filenames` to it.
