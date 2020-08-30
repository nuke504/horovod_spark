import os
import zipfile
import urllib.request
import io
import h5py
import shutil
import tempfile
import atexit
import pyarrow as pa

import pandas as pd
import re
import string
import unicodedata

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.backend as K

from pyspark import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType 
from pyspark.sql.functions import udf

import horovod.tensorflow.keras as hvd
from horovod.spark.task import get_available_devices

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset

DATA_SET_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
DATA_PATH = 'ml-100k/ml-100k'
PARQUET_PATH = 'parquet'
PETASTORM_HDFS_DRIVER = 'libhdfs'
NUM_PROC = 4
ALL_LETTERS = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
N_LETTERS = len(ALL_LETTERS) + 1
CHECKPOINT_FILE_NAME = 'checkpoint.h5'
DISCOVERY_SCRIPT = 'get_gpu_resources.sh'

def download_data_write_to_parquet(spark, data_path = DATA_PATH, parquet_path = PARQUET_PATH):
    '''
    Downloads the movielens dataset and converts all the dataframes to spark dataframes
    Writes the spark dataframes to 
    Deletes original zip file

    inputs:
        String data_path = DATA_PATH: Default path when movie lens dataset is unzipped
        String parquet_path = PARQUET_PATH: Path to store parquet files in

    Return:
        NIL
    '''
    if os.path.exists(os.path.join(os.getcwd(),DATA_PATH)):
        print('Data already downloaded')
    else:
        urllib.request.urlretrieve(DATA_SET_URL, 'ml-100k.zip')
        with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
            zip_ref.extractall('ml-100k')
        os.remove('ml-100k.zip')
        print('Data downloaded')

    df_genre = pd.read_csv(os.path.join(data_path,'u.genre'), sep = '|', header=None)
    df_genre.columns = ['genre','genre_id']

    df_movies = pd.read_csv(os.path.join(data_path,'u.item'), sep = '|', encoding='ISO-8859-1', header=None)
    df_movies.columns = ['movie_id','movie_title','release_date','video_release_date','imdb_url'] + df_genre.genre.values.tolist()
    df_movies = df_movies.drop(columns = ['video_release_date'])

    df_ratings = pd.read_table(os.path.join(data_path,'u.data'), header=None)
    df_ratings.columns = ['user_id','movie_id','rating','timestamp']

    df_users = pd.read_table(os.path.join(data_path,'u.user'), sep = '|', encoding='ISO-8859-1', header=None)
    df_users.columns = ['user_id','age','gender','occupation','zipcode']

    movies_schema = StructType([StructField("movie_id", IntegerType(), True),StructField("movie_title", StringType(), True),StructField("release_date", StringType(), True),StructField("imdb_url", StringType(), True)] + \
                      [StructField(genre, IntegerType(), True) for genre in df_genre.genre.values.tolist()])

    spark.createDataFrame(df_genre).write.parquet(os.path.join(parquet_path,'genre.parquet'),mode='overwrite')
    spark.createDataFrame(df_movies, schema = movies_schema).write.parquet(os.path.join(parquet_path,'movies.parquet'),mode='overwrite')
    spark.createDataFrame(df_ratings).write.parquet(os.path.join(parquet_path,'ratings.parquet'),mode='overwrite')
    spark.createDataFrame(df_users).write.parquet(os.path.join(parquet_path,'users.parquet'),mode='overwrite')

def get_write_parquet_avg_movie_rating(spark, parquet_path = PARQUET_PATH):
    '''
    Reads the movielens parquet files in parquet_path and get the average movie rating using sparkSQL
    Also makes slight edits to the movie title
    Entire function runs in spark
    inputs:
        String parquet_path = PARQUET_PATH: Path to store parquet files in

    Return:
        NIL
    '''

    spark.read.parquet(os.path.join(parquet_path,'genre.parquet')).createOrReplaceTempView("genre")
    spark.read.parquet(os.path.join(parquet_path,'movies.parquet')).createOrReplaceTempView("movies")
    spark.read.parquet(os.path.join(parquet_path,'ratings.parquet')).createOrReplaceTempView("ratings")
    spark.read.parquet(os.path.join(parquet_path,'users.parquet')).createOrReplaceTempView("users")

    query = """
        SELECT
            r.movie_id,
            m.movie_title,
            AVG(r.rating)/5.0 AS avg_rating,
            COUNT(r.rating) AS number_ratings
        FROM
            ratings AS r
        LEFT JOIN
            movies AS m
        ON
            r.movie_id = m.movie_id
        GROUP BY
            1,2
    """

    df_retrieved = spark.sql(query)
    regex_movie_title = udf(lambda x: unicodeToAscii(re.sub(r'\((\d+)\)','',x).lower(),ALL_LETTERS), StringType())
    df_retrieved = df_retrieved.withColumn('new_movie_title',regex_movie_title('movie_title'))
    df_retrieved = df_retrieved.select("new_movie_title", "avg_rating")
    df_train, df_validation, df_test = df_retrieved.randomSplit([0.7, 0.2, 0.1], seed=12345)
    
    train_rows = df_train.count()
    val_rows = df_validation.count()
    test_rows = df_test.count()

    df_train.write.parquet('%s/train_df.parquet' % PARQUET_PATH, mode='overwrite')
    df_validation.write.parquet('%s/val_df.parquet' % PARQUET_PATH, mode='overwrite')
    df_test.write.parquet('%s/test_df.parquet' % PARQUET_PATH, mode='overwrite')

    return train_rows, val_rows, test_rows

def serialize_model(model):
    """Serialize model into byte array."""
    bio = io.BytesIO()
    with h5py.File(bio) as f:
        model.save(f)
    return bio.getvalue()


def deserialize_model(model_bytes, load_model_fn):
    """Deserialize model from byte array."""
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio) as f:
        return load_model_fn(f)


def get_model():
    '''
    Returns a simple LSTM model

    '''
    encoder = get_encoder()
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())

    return model

def unicodeToAscii(s, all_letters = ALL_LETTERS):
    '''
    Helper function to convert unicode to ASCII string
    inputs:
        String s: String to convert to ASCII string
        String all_letters: A string of all the letters and symbols
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def get_encoder(all_letters = ALL_LETTERS):
    return tfds.features.text.SubwordTextEncoder(ALL_LETTERS)

def encode(text_tensor, label):
    '''
    Converts a text_tensor into an index tensor where each number represents the position of the character in the vocab string
    inputs:
        tf.tensor text_tensor: TF Tensor storing a string
        tf.tensor label: y for the text tensor

    return:
        tf.tensor encoded_text: index tensor
        tf.tensor label: label tensor
    '''
    encoder = get_encoder()

    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    '''
    A function to map encode to the tensor dataset
    inputs:
        tf.tensor text: TF Tensor storing a string
        tf.tensor label: y for the text tensor
    return:
        tf.tensor encoded_text: index tensor
        tf.tensor label: label tensor
    '''
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.float64))

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually: 
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

def train_fn(model_bytes, batch_size, epochs, train_rows, val_rows, warmup_epochs = 5, parquet_path = PARQUET_PATH, checkpoint_file_name = CHECKPOINT_FILE_NAME):
    '''
    Hovorod training function
    Can be used standalone or passed into a hovorod spark distributor
    NOTE: This function should run withou any bugs by itself
    
    inputs:
        io.bytes model_bytes: A serialised compiled keras model containing the model, optimiser, etc
        int batch_size: Batch size
        int epochs: # epochs
        int train_rows: # rows in training set
        int val_rows: # rows in val set
        int warmup_epochs: see callbacks section
        string parquet_path: path to training, validation and test parquet files
        string checkpoint_file_name: name of checkpoint file
    
    return:
        dict history: dictionary containing the history of loss and metrics
        io.bytes best_model_bytes: best model, serialised
    '''
    # Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
    # with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
    # functions like `deserialize_model` which are implemented at the top level.
    # See https://jira.apache.org/jira/browse/ARROW-3346
    pa    

    # Horovod: initialize Horovod inside the trainer.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process), if GPUs are available.
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices is not None:
        tf.config.set_visible_devices(physical_devices, 'GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # Horovod: restore from checkpoint, use hvd.load_model under the hood.
    model = deserialize_model(model_bytes, hvd.load_model)

    # Horovod: adjust learning rate based on number of processes.
    scaled_lr = K.get_value(model.optimizer.lr) * hvd.size()
    K.set_value(model.optimizer.lr, scaled_lr)

    # Horovod: print summary logs on the first worker.
    verbose = 2 if hvd.rank() == 0 else 0

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, initial_lr=scaled_lr, verbose=verbose),

        # Reduce LR if the metric is not improved for 10 epochs, and stop training
        # if it has not improved for 20 epochs.
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', patience=10, verbose=verbose),
        tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', mode='min', patience=20, verbose=verbose),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # Model checkpoint location.
    ckpt_dir = tempfile.mkdtemp()
    ckpt_file = os.path.join(ckpt_dir, checkpoint_file_name)
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_file, monitor='val_mean_squared_error', mode='min',
                                                            save_best_only=True))

    # Make Petastorm readers.
    with make_batch_reader('file://'+os.path.join(os.getcwd(),parquet_path,'train_df.parquet'), num_epochs=None,
                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                           hdfs_driver=PETASTORM_HDFS_DRIVER) as train_reader:
        with make_batch_reader('file://'+os.path.join(os.getcwd(),parquet_path,'val_df.parquet'), num_epochs=None,
                               cur_shard=hvd.rank(), shard_count=hvd.size(),
                               hdfs_driver=PETASTORM_HDFS_DRIVER) as val_reader:
            # Convert readers to tf.data.Dataset.
            train_ds = make_petastorm_dataset(train_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .shuffle(int(train_rows / hvd.size())) \
                .map(lambda x: encode_map_fn(x.new_movie_title,x.avg_rating)) \
                .padded_batch(batch_size)

            val_ds = make_petastorm_dataset(val_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .map(lambda x: encode_map_fn(x.new_movie_title,x.avg_rating)) \
                .padded_batch(batch_size)

            history = model.fit(train_ds,
                                validation_data=val_ds,
                                steps_per_epoch=int(train_rows / batch_size / hvd.size()),
                                validation_steps=int(val_rows / batch_size / hvd.size()),
                                callbacks=callbacks,
                                verbose=verbose,
                                epochs=epochs)

    # Dataset API usage currently displays a wall of errors upon termination.
    # This global model registration ensures clean termination.
    # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
    globals()['_DATASET_FINALIZATION_HACK'] = model

    if hvd.rank() == 0:
        with open(ckpt_file, 'rb') as f:
            return history.history, f.read()


def predict_fn(model_bytes):
    '''
    A function to be applied row-wise to the pyspark dataframe for prediction
    Entirely run in spark
    inputs:
        io.bytes model_bytes: A serialised compiled keras model containing the model, optimiser, etc
    '''
    def fn(rows):

        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices is not None:
            tf.config.set_visible_devices(physical_devices, 'GPU')

        # Restore from checkpoint.
        model = deserialize_model(model_bytes, tf.keras.models.load_model)
        encoder = get_encoder()

        # Perform predictions.
        for row in rows:
            fields = row.asDict().copy()
            
            input_tensor = tf.constant([encoder.encode(row.new_movie_title)])
            fields['avg_predicted_rating'] = model.predict_on_batch(input_tensor).item()
            yield Row(**fields)

    return fn

def set_gpu_conf(conf):
        # This config will change depending on your cluster setup.
        #
        # 1. Standalone Cluster
        # - Must configure spark.worker.* configs as below.
        #
        # 2. YARN
        # - Requires YARN 3.1 or higher to support GPUs
        # - Cluster should be configured to have isolation on so that
        #   multiple executors don’t see the same GPU on the same host.
        # - If you don’t have isolation then you would require a different discovery script
        #   or other way to make sure that 2 executors don’t try to use same GPU.
        #
        # 3. Kubernetes
        # - Requires GPU support and isolation.
        # - Add conf.set(“spark.executor.resource.gpu.discoveryScript”, DISCOVERY_SCRIPT)
        # - Add conf.set(“spark.executor.resource.gpu.vendor”, “nvidia.com”)
        conf = conf.set("spark.test.home", os.environ.get('SPARK_HOME'))
        conf = conf.set("spark.worker.resource.gpu.discoveryScript", DISCOVERY_SCRIPT)
        conf = conf.set("spark.worker.resource.gpu.amount", 1)
        conf = conf.set("spark.task.resource.gpu.amount", "1")
        conf = conf.set("spark.executor.resource.gpu.amount", "1")
        return conf