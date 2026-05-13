# RLforMH

## Initial setup
### Clone the repo
```
git clone https://github.com/ElisabethHolm/RLforMH.git
cd RLforMH
```

### Create a virtual environment and install requirements
```
python3.10 -m venv cs224r
source cs224r/bin/activate
pip install -r requirements.txt
```

## Dataset creation
### Download the original StudentLife dataset
https://www.kaggle.com/datasets/dartweichen/student-life/data 

Put it into a folder studentLifeDataset so you have
RLforMH
├── studentLifeDataset
│   ├── app_usage
│   ├── calendar
│   ├── ...
│   └── survey
├── dataset_prep
└── ...

### Build the aggregated dataset
```
python build_aggregated_data.py
```

### Add in rewards, actions, etc to prep for training
```
python prepare_rl_dataset.py
```

## Training
## Run training on daily_studentlife.csv

