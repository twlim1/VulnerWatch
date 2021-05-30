# Python notebook:

## Dependencies
1. All notebooks are written and tested on Google Colab. They can be modified to run locally.
2. All dependencies are self-contained and installed in the notebook

## Filename and usage
1. data/data_download_cve.ipynb
* Notebook to download CVE data using REST API from NVD website. The downloaded data is saved in json format

2. data/data_download_cpe.ipynb
* Notebook to download CPE data using REST API from NVD website. The downloaded data is saved in json format
* CPE data is not used in the project, but could be useful to enhance prediction.

3. data/cve_train_test_split.ipynb
* Notebook to read in cve json file and split 80-20 for train/test data. Output data is in csv format

4. models/metric_pr.ipynb
* Notebook to read in cve training data in csv format. Then training data is split for train/validation set
* Train dataset is used to fine tune bert-base-uncased model to predict "Privileged Required" metric
* Validation dataset is used to measure model performance, including Accuracy, MCC, Precision, Recall and F1
* The same notebook can be modified slightly to predict other metrics

5. models/metric_ui.ipynb
* Notebook to read in cve training data in csv format. Then training data is split for train/validation set
* Train dataset is used to fine tune bert-base-uncased model to predict "User Interaction" metric
* Validation dataset is used to measure model performance, including Accuracy, MCC, Precision, Recall and F1
* The same notebook can be modified slightly to predict other metrics

6. models/cvss_prediction.ipynb
* Notebook to load all 8 pretrained/fine-tuned BERT models and make prediction on all 8 metrics
* The final CVSS scores are calculated based on the predicted values of metrics.
* Confidence scores are calculated
* MSE, MAE, R2 are measured. Can be fine-tuned for different confidence threshold

7. models/semantic_similarity.ipynb
* Notebook to find the CVE description that most similar to the input text by the user
* Code is not use in final project. It could be useful for future enhancement

8. models/createModels.ipynb
* Notebook that takes params to train all 8 models using models.py module
