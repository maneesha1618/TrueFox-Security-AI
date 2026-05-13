# Test Data — data_sample/

This folder should contain test video clips for running inference.

## Required files
- `fight_1.avi`, `fight_2.avi`, `fight_3.avi` — real fight clips
- `nonfight_1.avi`, `nonfight_2.avi`, `nonfight_3.avi` — normal clips
- `weapon_test_video.avi` — weapon detection test video

## Download test clips

**Google Drive (ready to use clips):**
[Download test clips](https://drive.google.com/drive/folders/153WFcFicN7i4pvH5pCzNmpa0Bo93LboD?usp=drive_link)

**Original dataset source:**
https://www.kaggle.com/datasets/vulamnguyen/rwf2000

## How to use
After downloading, place all .avi files in this folder:

```
data_sample/
├── fight_1.avi
├── fight_2.avi
├── fight_3.avi
├── nonfight_1.avi
├── nonfight_2.avi
├── nonfight_3.avi
└── weapon_test_video.avi
```

Then run:
```bash
python main.py --source data_sample/fight_1.avi --save
```

