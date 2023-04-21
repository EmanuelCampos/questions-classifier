# Question Classifier (WIP)
Learning repo to train a model using StackOverflow data set questions and classify questions as good questions or bad questions

# How to run
1. install pip packages
2. Download dataset from: https://www.kaggle.com/datasets/stackoverflow/rquestions?resource=download and move to `/src/assets/questions.csv`
2. Pré-process the dataset on `src/assets/questions.csv` running on root: `python3 ./src/pre_process.py`
3. Train and run the model `python ./src/train.py`
