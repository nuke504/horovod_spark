import argparse
import os

from utils import download_data_write_to_parquet, get_write_parquet_avg_movie_rating, get_model, serialize_model, train_fn, set_gpu_conf, predict_fn

import tensorflow as tf

from pyspark import SparkConf
from pyspark.sql import SparkSession

import horovod.tensorflow.keras as hvd
import horovod.spark

def get_options():
    parser = argparse.ArgumentParser(description='Keras Spark3 Rossmann Run Example',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--processing-master',
                        help='spark cluster to use for light processing (data preparation & prediction).'
                            'If set to None, uses current default cluster. Cluster should be set up to provide'
                            'one task per CPU core. Example: spark://hostname:7077')
    parser.add_argument('--training-master', default='local-cluster[2,1,1024]',
                        help='spark cluster to use for training. If set to None, uses current default cluster. Cluster'
                            'should be set up to provide a Spark task per multiple CPU cores, or per GPU, e.g. by'
                            'supplying `-c <NUM_GPUS>` in Spark Standalone mode. Example: spark://hostname:7077')
    parser.add_argument('--num-proc', type=int, default=2,
                        help='number of worker processes for training, default: `spark.default.parallelism`')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--data-dir', default='file://' + os.path.join(os.getcwd(), 'parquet'),
                        help='location of data on local filesystem (prefixed with file://) or on HDFS')
    parser.add_argument('--local-prediction-csv', default='prediction.csv',
                        help='output submission predictions CSV on local filesystem (without file:// prefix)')
    parser.add_argument('--local-checkpoint-file', default='checkpoint.h5',
                        help='model checkpoint on local filesystem (without file:// prefix)')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    
    args = get_options()

    # 1. Create Spark session for data preparation.
    conf = SparkConf().setAppName('data_prep').set('spark.sql.shuffle.partitions', '16')
    if args.processing_master:
        conf.setMaster(args.processing_master)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    download_data_write_to_parquet(spark, parquet_path = args.data_dir)

    train_rows, val_rows, _ = get_write_parquet_avg_movie_rating(spark, parquet_path = args.data_dir)

    spark.stop()

    # Do not use GPU for the session creation.
    tf.config.experimental.set_visible_devices([], 'GPU')

    model = get_model()

    # 2. Horovod: add Distributed Optimizer.
    opt = tf.keras.optimizers.Adam(lr=args.learning_rate, epsilon=1e-3)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(opt, loss = tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    model_bytes = serialize_model(model)

    # Create Spark session for training.
    conf = SparkConf().setAppName('training')
    # if args.training_master:
    #     conf.setMaster(args.training_master)
    conf = set_gpu_conf(conf)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    history, best_model_bytes = horovod.spark.run(train_fn, args=(model_bytes, args.batch_size, args.epochs, train_rows, val_rows), num_proc=args.num_proc, verbose=2)[0]

    print(f"Best MSE: {min(history['val_mean_squared_error'])}")
    # Write checkpoint.
    with open(args.local_checkpoint_file, 'wb') as f:
        f.write(best_model_bytes)
    print(f'Written checkpoint to {args.local_checkpoint_file}')
    spark.stop()

    # 3. Prediction
    conf = SparkConf().setAppName('prediction') \
        .setExecutorEnv('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH')) \
        .setExecutorEnv('PATH', os.environ.get('PATH'))

    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    pred_df = spark.read.parquet(os.path.join(args.data_dir, 'test_df.parquet')).rdd.mapPartitions(predict_fn(best_model_bytes)).toDF()
    pred_df = pred_df.orderBy('avg_predicted_rating', ascending=False).toPandas()
    pred_df.to_csv(os.path.join(os.getcwd(),args.local_prediction_csv))
    # pred_df.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save(os.path.join(os.getcwd(),args.local_prediction_csv))
    spark.stop()


    
    
