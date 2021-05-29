# Data:

1. raw/cve.json
* Raw data download from NVD website using ../models/data/data_download_cve.ipynb

1. interim/cve_train.csv
* Train dataset split from cve.json using ../models/data/cve_train_test_split.ipynb

1. interim/cve_test.csv
* Test dataset split from cve.json using ../models/data/cve_train_test_split.ipynb

1. processed/cve_test_prediction_results.json
* Predicted values of test dataset using all 8 BERT models