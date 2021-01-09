import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import shutil
import collections
import math

CHECK_ROOT_DIR = '../HOME/MODELS/'

def _shuffleMultiArray(tensors):
    """ Shuffle two numpy array simultaneously"""
    perm = np.random.permutation(np.arange(tensors[0].shape[0]))
    tensors_perm = [None] * len(tensors)
    for i in range(len(tensors)):
        tensors_perm[i] = tensors[i][perm]
    return tensors_perm

def batchDispenser(tensors, batch_size, epochs, shuffle=True, tqdm_pb=False):
    """ Factory that produce batches for the training process"""
    n = len(tensors[0])

    iterator = range(epochs)
    if tqdm_pb:
        iterator = tqdm(iterator)

    for i, epoch in enumerate(iterator):
        for j, batch_start in enumerate(range(0, n, batch_size)):
            batch = [t[batch_start:batch_start + batch_size] for t in tensors]
            yield i, j, batch
        if shuffle:
            tensors = _shuffleMultiArray(tensors)


def makeDataset(data, epchos_num, batch_size, shuffle_buffer_size=10000, drop_remainder=True):
    """ Build a tensorflow.Dataset object with the specified batch size, epchos number and shuffle operation """
    dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.repeat(epchos_num)
    
        
    if drop_remainder:
        tf.logging.info("Drop remainder feature is actived in batch dispenser")
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset

def makeBatchTensor(data, epchos_num, batch_size, shuffle_buffer_size=10000, with_tqdm=False, drop_remainder=True):
    """ Main training loop iterator function """
    dataset = makeDataset(data, epchos_num, batch_size, shuffle_buffer_size=shuffle_buffer_size, drop_remainder=drop_remainder)
    dataset_iterator = dataset.make_one_shot_iterator()
    
    if isinstance(data, tuple):
        n = data[0].shape[0]
    else:
        n = data.shape[0]
        
    batch4epochs = math.ceil(n / batch_size)
    number_of_batch = batch4epochs * epchos_num

    batch = dataset_iterator.get_next()

    return batch, number_of_batch

TRAIN, PREDICT, EVAL = tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL
RAW = '#RAW'


def getSessionConfing():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def accuracy(logits, y):
    return tf.contrib.metrics.accuracy(tf.reshape(y, (-1,)), tf.argmax(logits,  1, output_type=tf.int32))

class RawModel:
    SUMMARY_IMGS_MAX = 3
    
    def __init__(self, name, optimizer=tf.train.AdamOptimizer, learning_rate=0.001, params={}, save_summary=True, model_dir=None):

        self.name = name

        if model_dir:
            self.model_dir_path = model_dir
        else:
            self.model_dir_path = CHECK_ROOT_DIR


        self.model_dir = os.path.join(self.model_dir_path, name + '/') 

        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.hparams = params

        self.hparams.update({'learning_rate':learning_rate})
        
        self.save_summary = save_summary
        
        self.fires = []
        self.fires_mean = None
        
    def _fireMean(self):
        self.fires_mean = [tf.reduce_mean(fire, 1) for fire in fires]

    def appendFire(self, fire):
        self.fires.append(fire)

    def model_function(self, mode, params, constants=None, **kargs):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            y_ = self.build_model(mode, params, constants=constants, **kargs)

            if mode == RAW:
                return y_

            if mode == PREDICT:
                self.predict_ops = self.setup_predict(features, y, y_)
                return self.predict_ops

            if mode == TRAIN:
                

                # make loss and training ops
                self.loss = self.setup_loss(y_, **kargs)
                self.train_op = self.build_train_op(params)
                return self.train_op

    def makeSession(self, newvar_list=None):
        """ Create and return a tensorflow session with all model's variables initialized """

        config = getSessionConfing()
        sess = tf.Session(config=config)

        if os.path.isdir(self.model_dir):
            
            if newvar_list:
                vars_init = [v.initializer for v in newvar_list]
                sess.run(vars_init)
                
                variables = list( set(tf.trainable_variables()) - set(newvar_list) )
                
                self.saver = tf.train.Saver(variables)
                self.saver.restore(sess, self.model_dir)
                
            else:
                self.saver = tf.train.Saver()

                tf.logging.info('CHECKPOINT RESTORED')
                self.saver.restore(sess, self.model_dir)
        else:
            self.saver = tf.train.Saver()

            tf.logging.info('INIT FROM SCRATCH. Model\'s directory: %s' % self.model_dir)
            os.mkdir(self.model_dir)
            init = tf.global_variables_initializer()
            sess.run(init)
        
        sess.run(tf.tables_initializer())
        
        return sess

    def reset(self):
        """ Delete from the filesystem all model checkpoints """
        tf.reset_default_graph()
        try:
            shutil.rmtree(self.model_dir)
            print('Model folder %s has been removed' % self.model_dir)
        except FileNotFoundError:
            print('Model reset fail! File not found: %s' % self.model_dir)
        except OSError:
            print('Is not possible remove the folder: %s try manually' % self.model_dir)
            
    def clone(self, new_model_tree):
        try:
            shutil.copytree(self.model_dir, new_model_tree)
            print('Model folder %s has been copied to %s' % (self.model_dir, new_model_tree))
        except:
            print('Is not possible copy the folder: %s try manually' % self.model_dir)


    def build_train_op(self, params):

        lr = self.getLearningRate(params)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = self.optimizer(learning_rate=lr).minimize(self.loss, global_step=self.global_step)

        return [train_op]
    
    
    def train(self, data, epochs, batch_size, summary_steps_number, constants={}, restartGraphAfterTraining=True, use_tqdm=True, max_iterations=None):

        if not isinstance(data, dict):
            raise NotImplementedError()

        data_key = sorted(list(data.keys()))
        data_list = tuple( [data[k] for k in data_key] )
        batch, number_of_batches = makeBatchTensor(data_list, epochs, batch_size, shuffle_buffer_size=10000)
        data_tf = {data_key[i]:batch[i] for i in range(len(batch))}
        
        #self.hparams.update(batch_size=batch_size)

        constants_tf = {}
        if constants:
            for k, v in constants.items():
                constants_tf[k] = tf.constant(v)

        train_ops = self.model_function(TRAIN, self.hparams, constants_tf, **data_tf)
           
        merged = tf.summary.merge_all()
        train_ops_and_summary = [merged] + train_ops
        
        number_of_batches += 1
        summary_intervall = number_of_batches // summary_steps_number
        with self.makeSession() as sess:
            
            tf.logging.info( 'Number of training steps: %s' % number_of_batches )
            
            if self.save_summary:
                train_writer = tf.summary.FileWriter(self.model_dir, sess.graph)
                #tf.logging.info( 'Number of summary steps: %s' % (number_of_batches // summary_intervall) )

            try:
                if max_iterations:
                    number_of_batches = max_iterations
                    
                batch_range = tqdm( range(number_of_batches) ) if use_tqdm else range(number_of_batches)
                for batch_i in batch_range:
                    #get global step
                    i = self.global_step.eval(sess)
                    
                    if max_iterations and i >= max_iterations:
                        break
                    
                    if self.save_summary and i % summary_intervall == 0:                        
                        summary, *_ = self.training_apply(sess, train_ops_and_summary)
                        train_writer.add_summary(summary, i)
                    else:
                        _ = self.training_apply(sess, train_ops)

            except tf.errors.OutOfRangeError:
                ...
            self.saver.save(sess, self.model_dir)
            tf.logging.info('CHECKPOINT SAVED')
                
        if restartGraphAfterTraining:
            tf.logging.info('DEFAULT GRAPH RESET')
            tf.reset_default_graph()
            
            
    def train_on_dynamic_dataset(self, number_of_batches, batch_size, summary_steps_number, constants={}, restartGraphAfterTraining=True, use_tqdm=True):
        
        constants_tf = {}
        if constants:
            for k, v in constants.items():
                constants_tf[k] = tf.constant(v)
        
        train_ops = self.model_function(TRAIN, self.hparams, constants_tf)
           
        merged = tf.summary.merge_all()
        train_ops_and_summary = [merged] + train_ops
        
        summary_intervall = number_of_batches // summary_steps_number
        
        with self.makeSession() as sess:
            
            tf.logging.info( 'Number of training steps: %s' % number_of_batches )
            
            if self.save_summary:
                train_writer = tf.summary.FileWriter(self.model_dir, sess.graph)
                #tf.logging.info( 'Number of summary steps: %s' % (number_of_batches // summary_intervall) )

                batch_range = tqdm( range(number_of_batches) ) if use_tqdm else range(number_of_batches)
                for batch_i in batch_range:
                    #get global step
                    i = self.global_step.eval(sess)
                    
                    if self.save_summary and i % summary_intervall == 0:                        
                        summary, *_ = self.training_apply(sess, train_ops_and_summary)
                        train_writer.add_summary(summary, i)
                    else:
                        _ = self.training_apply(sess, train_ops)

            self.saver.save(sess, self.model_dir)
            tf.logging.info('CHECKPOINT SAVED')
                
        if restartGraphAfterTraining:
            tf.logging.info('DEFAULT GRAPH RESET')
            tf.reset_default_graph()
        

    def checkBatch(self, data, n, batch_size):
        
        data_key = sorted(list(data.keys()))
        data_list = tuple( [data[k] for k in data_key] )
        batch, number_of_batches = makeBatchTensor(data_list, n, batch_size, shuffle_buffer_size=10000)
        data_tf = {data_key[i]:batch[i] for i in range(len(batch))}
          
        out = [None] * n
        
        with tf.Session(config=getSessionConfing()) as sess:
            for i in range(n):
                out[i] = sess.run(batch)
                
        return out

    def training_apply(self, sess, train_ops):
        return sess.run(train_ops)

    def incrementGlobalStepOp(self):
        return tf.assign(self.global_step, self.global_step+1)

    def getLearningRate(self, params):
        return params['learning_rate']

    def setup_eval_metrics(self, x, y, y_):
        raise NotImplemented()

    def setup_loss(self, y_):
        raise NotImplemented()

    def setup_predict(self, y_):
        raise NotImplemented()

    def build_train_op(self, params):
        raise NotImplemented()

    def build_model(self, mode, params, constants):
        raise NotImplemented()
        
    def summary_image(self, name, img):
        tf.summary.image(name, img, max_outputs=self.SUMMARY_IMGS_MAX)
    
    def exportHubModule(self, path, hub_model_fn):
        import tensorflow_hub as hub
        
        tf.reset_default_graph()

        spec = hub.create_module_spec(hub_model_fn)
        module = hub.Module(spec, name=self.name, trainable=True)
        
        with self.makeSession() as sess:
            module.export(path, sess)
