## A Toy Example that utilises Horovod on Spark for distributed deep learning training
Uses code adapted from Horovod's spark tutorial for tabular data, see https://github.com/horovod/horovod/blob/master/examples/keras_spark3_rossmann.py

All sparkconf can be edited in run.py

### Instructions to Run
To run the file:
python run.py

### Requirements
1. Latest Horovod
2. Latest Tensorflow
3. Tensorflow Datasets
4. Spark 2.4.6 (not 3.0.0)
5. Petastorm
6. Java 8 (not 11)

### Process
Each step uses a separate SparkConf and Spark session
1. Download data and write to parquet files
2. Read from parquet and train the LSTM
3. Predict on test set using trained model and write results to local