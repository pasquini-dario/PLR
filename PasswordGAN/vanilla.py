import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PasswordGAN.FAIL import rawModel
from PasswordGAN.FAIL.rawModel import TRAIN
from tqdm import tqdm
import math


UNIFORM = 'uniform'
NORMAL = 'normal'
LEARN_NORMAL = 'lnormal'
G_name = 'G'
D_name = 'D'

def makeDataset(data, epchos_num, batch_size, shuffle_buffer_size=10000, drop_remainder=True):
    """ Build a tensorflow.Dataset object with the specified batch size, epchos number and shuffle operation """
    dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.repeat(None)
        
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

class Vanilla(rawModel.RawModel):
        
    def train(self, data, epochs, batch_size, summary_steps_number, constants={}, restartGraphAfterTraining=True, use_tqdm=True):

        if not isinstance(data, dict):
            raise NotImplementedError()

        data_key = sorted(list(data.keys()))
        data_list = tuple( [data[k] for k in data_key] )
        batch, _ = makeBatchTensor(data_list, epochs, batch_size, shuffle_buffer_size=10000)
        data_tf = {data_key[i]:batch[i] for i in range(len(batch))}
        
        constants_tf = {}
        if constants:
            for k, v in constants.items():
                constants_tf[k] = tf.constant(v)

        gen_train_op, disc_train_op = self.model_function(TRAIN, self.hparams, constants_tf, **data_tf)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print(update_ops)
        gen_train_op = tf.group([gen_train_op, update_ops])
        
        D_iters = self.hparams['D_iters']
        #  D consumes more batch for a whole training step
        number_of_batches = epochs
        
        hard_evaluation_interval = self.hparams['evaluation']['evaluation_fq']
           
        merged = tf.summary.merge_all()
        
        summary_intervall = number_of_batches // summary_steps_number
        with self.makeSession() as sess:
            
            tf.logging.info( 'Number of training steps: %s' % number_of_batches )
            
            if self.save_summary:
                train_writer = tf.summary.FileWriter(self.model_dir, sess.graph)

            if True:
                batch_range = tqdm( range(number_of_batches) ) if use_tqdm else range(number_of_batches)
                for batch_i in batch_range:
                    #get global step
                    i = self.global_step.eval(sess)
                    
                    if self.save_summary and i % summary_intervall == 0:                        
                        summary = sess.run(merged)
                        train_writer.add_summary(summary, i)
                    
                    if hard_evaluation_interval:
                        if i and i % hard_evaluation_interval == 0:
                            self.evaluation(sess)

                    # generator step
                    sess.run(gen_train_op)

                    # discriminator steps
                    for k in range(D_iters):
                        sess.run(disc_train_op)
                                                
            # save the model
            self.saver.save(sess, self.model_dir)
            tf.logging.info('CHECKPOINT SAVED')
                
        if restartGraphAfterTraining:
            tf.logging.info('DEFAULT GRAPH RESET')
            tf.reset_default_graph()
   
            
    def index_to_strings(self, index): 
        index = tf.argmax(index, axis=2)
        chars = self.char_map.lookup(index)
        strings = tf.reduce_join(chars, axis=1)
        return strings
    
    def build_model(self, mode, params, constants, x=None, z=None):
        
        is_training = rawModel.TRAIN == mode
        
        z_size = params['z_size']
        z_prior = params['z_prior']
        x_size = params['x_size']
        
        G_maker = params['G_maker']
        D_maker = params['D_maker']
        
        chars = params['chars']
        dict_size = len(chars)
        self.char_map = tf.contrib.lookup.index_to_string_table_from_tensor(chars)
        
        evaluation_batch_size = params['evaluation']['evaluation_batch_size']
        
        if z is None:
            batch_size = params['batch_size']
            z = tf.random_normal(dtype=tf.float32, shape=(batch_size, z_size), mean=0, stddev=z_prior)
            z_val = tf.random_normal(dtype=tf.float32, shape=(evaluation_batch_size, z_size), mean=0, stddev=z_prior)
            print(z)

                
        with tf.variable_scope(G_name, reuse=tf.AUTO_REUSE) as scope:
            G = G_maker(z, x_size, dict_size, is_training=is_training)
        self.G_vars = scope.trainable_variables()
        
        # generator for eval (with bigger batch-size)
        with tf.variable_scope(G_name, reuse=tf.AUTO_REUSE) as scope:
            self.G_val = G_maker(z_val, x_size, dict_size, is_training=False)

        self.passwords = self.index_to_strings(G)
        
        if is_training:
            # logging passwords sample
            tf.summary.text('G', self.passwords)
            
            x = self.transform_truedata(x)
            
            with tf.variable_scope(D_name, reuse=tf.AUTO_REUSE) as scope:
                D = D_maker(x, x_size, is_training=is_training)
                
            self.D_vars = scope.trainable_variables()
            
            with tf.variable_scope(D_name, reuse=tf.AUTO_REUSE):
                DG = D_maker(G, x_size, is_training=is_training)
                
        else:
            D = None
            DG = None

        return {
            'D' : D,
            'DG' : DG,
            'G' : G,
            'x':x,
            #'mu' : mu,
            #'sigma' : sigma,
        }     
    
    
    def transform_truedata(self, x):
        dict_size = len(self.hparams['chars'])
        x = tf.one_hot(x, dict_size)
        
        # label smoothing
        gamma = self.hparams['gamma']
        batch_size = self.hparams['batch_size']
        x_size = self.hparams['x_size']
        if gamma:
            x = x + (tf.random_uniform((batch_size, x_size, dict_size)) * gamma)
            # normalize
            x = x / tf.reduce_sum(x, 2, keep_dims=True)
            
        return x
    
    def setup_loss(self, y_, x, **kargs):

        D = y_['D']
        DG = y_['DG']
        G = y_['G']
        xhoe = y_['x']
        batch_size = self.hparams['batch_size']
        
        losses = self.wgangp_loss(
            G,
            xhoe,
            D,
            DG,
        )

        gen_cost, disc_cost, gradient_penalty = losses
        
        if 'sigma' in y_:
            sigma = y_['sigma']
            
            sigma_full = tf.matmul(sigma, sigma, transpose_b=True)[:,:,:,None]
            
            tf.summary.image('sigma_img', sigma_full)
            tf.summary.histogram('sigma', sigma)
            
        tf.summary.scalar('G_loss', gen_cost)
        tf.summary.scalar('D_loss', disc_cost)
        tf.summary.scalar('D_gp', gradient_penalty)    
        
        # setup evals
        self.setupEvaluation()

        return [gen_cost, disc_cost]
    
    def setupEvaluation(self):
        #evaluation
        # number of match on the test-set [no reps]
        test_match = tf.Variable(0., trainable=False)
        self.test_match_value = tf.placeholder(tf.float32)
        self.assign_test_match = test_match.assign(self.test_match_value)
        
        # number of match on the clean test-set [no reps no train intersection]
        test_match_clean = tf.Variable(0., trainable=False)
        self.test_match_value_clean = tf.placeholder(tf.float32)
        self.assign_test_match_clean = test_match_clean .assign(self.test_match_value_clean)
        
        # number of unique passwords
        unique = tf.Variable(0., trainable=False)
        self.unique_value = tf.placeholder(tf.float32)
        self.assign_unique = unique.assign(self.unique_value)
        
        tf.summary.scalar('test_match', test_match)
        tf.summary.scalar('test_match_clean', test_match_clean)
        tf.summary.scalar('unique', unique)
    
    def evaluation(self, sess):
        params = self.hparams['evaluation']
        generator_samples = params['generator_samples']
        test_set = params['test_set']
        test_set_clean = params['test_set_clean']
        batch_size = params['evaluation_batch_size']
        
        out = set()
        
        n_batch = math.ceil(generator_samples / batch_size)
        print(generator_samples, batch_size, n_batch)
        
        print("START EVAL....")
        
        p_op = self.G_val
        p_op = tf.argmax(p_op, axis=2, output_type=tf.int32)
        p_op = tf.cast(p_op, tf.uint8)
        
        for i in tqdm(range(n_batch)):
            ps = sess.run(p_op)
            for p in ps:
                out.add(p.tobytes())
            
        matches = len( out.intersection(test_set) ) / len(test_set)
        sess.run(self.assign_test_match, {self.test_match_value:matches})
        
        matches_clean = len( out.intersection(test_set_clean) ) / len(test_set_clean)
        sess.run( self.assign_test_match_clean, {self.test_match_value_clean:matches_clean} )
        
        unique = len(out) / (n_batch * batch_size)
        sess.run(self.assign_unique, {self.unique_value:unique})
        
        print("....END EVAL")

    
    def build_train_op(self, params):
        
        lr = self.getLearningRate(params)
        
        beta1 = self.hparams['beta1']
        beta2 = self.hparams['beta2']
        
        train_op_G = self.optimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(
            self.loss[0],
            global_step=self.global_step,
            var_list=self.G_vars
        )
        
        train_op_D = self.optimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(
            self.loss[1],
            var_list=self.D_vars
        )

        return [train_op_G, train_op_D]
    
    
    def wgangp_loss(self, fake_data, real_data, disc_real, disc_fake, LAMBDA=10):
        D_maker = self.hparams['D_maker']
        batch_size = self.hparams['batch_size']
        x_size = self.hparams['x_size']

        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        alpha = tf.random_uniform(
            shape=[batch_size, 1, 1], 
            minval=0.,
            maxval=1.
        )

        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)

        with tf.variable_scope(D_name, reuse=tf.AUTO_REUSE):
            D = D_maker(interpolates, x_size, is_training=True)

        gradients = tf.gradients(D, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = LAMBDA * tf.reduce_mean((slopes-1.)**2)
        disc_cost += gradient_penalty

        losses = [gen_cost, disc_cost, gradient_penalty]

        return losses
    
    
    def hub_function(self):
        
        params = self.hparams
        z_size = params['z_size']
        z_prior = params['z_prior']
        x_size = params['x_size']
        G_maker = params['G_maker']
        D_maker = params['D_maker']
        chars = params['chars']
        dict_size = len(chars)
        self.char_map = tf.contrib.lookup.index_to_string_table_from_tensor(chars)

        z = tf.placeholder(shape=(None, z_size), dtype=tf.float32)
        is_training = tf.placeholder_with_default(False, shape=())
        
        inputs = {
            'default':z,
            'is_training':is_training,
        }
        ## data inference given z [G]
        with tf.variable_scope(G_name, reuse=tf.AUTO_REUSE) as scope:
            G, logits = G_maker(z, x_size, dict_size, is_training=is_training, with_logits=True)
        password = self.index_to_strings(G)
        
        outputs = {
            'default' : password,
            'p' : G,
            'logits' : logits,
        }
        hub.add_signature('latent_to_data', inputs=z, outputs=outputs)

        
