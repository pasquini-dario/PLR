import tensorflow as tf
import tensorflow_hub as hub
from estimator import Estimator
import numpy as np
import myPickle, os 
import math
import h5py

def multiSoftmaxCrossEntropy(label, logits):
    num_classes = logits.shape.as_list()[-1]
    assert num_classes == label.shape.as_list()[-1]
    logits = tf.reshape(logits, [-1, num_classes])
    label = tf.reshape(label, [-1, num_classes])
    return tf.losses.softmax_cross_entropy(label, logits)

def multiSoftmaxCrossEntropyW(label, logits, w):
    batch_size, char_num, num_classes = logits.shape.as_list()
    assert num_classes == label.shape.as_list()[-1]
    logits = tf.reshape(logits, [-1, num_classes])
    label = tf.reshape(label, [-1, num_classes])
    print(label, logits)
    loss =  tf.losses.softmax_cross_entropy(label, logits, reduction=tf.losses.Reduction.NONE)
    loss = tf.reshape(loss, (-1, char_num))
    loss = tf.reduce_sum(loss, 1) * w
    loss = tf.reduce_mean(loss) 
    return loss

def index_accuracy(a, b):
    eq = tf.cast(tf.equal(a, b), tf.float32)
    acc = tf.reduce_mean( eq )
    return acc

ENC = 'enc'
DEC = 'dec'

class AE(Estimator):

    def exportHubModule(self, path, input_shape, trainable=False):
        import tensorflow_hub as hub

        self.input_shape = input_shape

        spec = hub.create_module_spec(self.hub_model_fn)
        module = hub.Module(spec, name=Estimator.NAME, trainable=trainable)
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.estimator.latest_checkpoint())
            sess.run(tf.tables_initializer())
            module.export(path, sess)

    def hub_model_fn(self):

        training = False

        xph = tf.placeholder(tf.int32, shape=(None, *self.input_shape))
        
        features = {'x':xph}
        out = self.setup_predictions(features, training)
        out.update({'default':out['p']})

        hub.add_signature(inputs=xph, outputs=out)

        ###################
        ls = self.hparams['latent_size']

        zph = tf.placeholder(tf.float32, shape=(None, ls))
        
        out_shape = out['x'].shape.as_list()[1:]
        out = self.setup_predictions_latent(zph, out_shape, training)
        out.update({'default':out['p']})

        hub.add_signature(name='latent', inputs=zph, outputs=out)
        
    def index_to_strings(self, index): 
        index = tf.argmax(index, axis=2)
        chars = self.char_map.lookup(index)
        strings = tf.reduce_join(chars, axis=1)
        return strings

    def setup_predictions(self, features, training):

        char_num = self.hparams['char_num']
        enc_arch = self.hparams['enc_arch']
        dec_arch = self.hparams['dec_arch']
        
        ## CMAP
        latent_size = self.hparams['latent_size'] 
        chars = self.hparams['chars']
        self.char_map = tf.contrib.lookup.index_to_string_table_from_tensor(chars)
        ##
        
        x = features['x']
        x = tf.one_hot(x, char_num)

        with tf.variable_scope(ENC, reuse=tf.AUTO_REUSE) as scope:
            z = enc_arch(x, latent_size, training)
        
        out_shape = x.shape.as_list()[1:]

        with tf.variable_scope(DEC, reuse=tf.AUTO_REUSE) as scope:
            logits = dec_arch(z, out_shape, training)
        
        loss_id = self.hparams['loss_id']
        if loss_id == 0:
            p = tf.nn.softmax(logits, 2)
        elif loss_id == 1:
            p = tf.nn.sigmoid(logits)
            
        prediction = tf.argmax(p, 2, output_type=tf.int32)
        prediction_string = self.index_to_strings(p)
        
        out = {'logits':logits, 'p':p, 'prediction':prediction, 'x':x, 'z':z, 'prediction_string':prediction_string}
        
        if 'r' in features:
            out.update({'r':features['r']})

        return out
    
    def setup_predictions_latent(self, z, out_shape, training):

        char_num = self.hparams['char_num']
        dec_arch = self.hparams['dec_arch']
        latent_size = self.hparams['latent_size']          
        
        ## CMAP
        latent_size = self.hparams['latent_size'] 
        chars = self.hparams['chars']
        self.char_map = tf.contrib.lookup.index_to_string_table_from_tensor(chars)
        ##
        
        with tf.variable_scope(DEC, reuse=tf.AUTO_REUSE) as scope:
            logits = dec_arch(z, out_shape, training)
        
        loss_id = self.hparams['loss_id']
        if loss_id == 0:
            p = tf.nn.softmax(logits, 2)
        elif loss_id == 1:
            p = tf.nn.sigmoid(logits)
            
        prediction = tf.argmax(p, 2, output_type=tf.int32)
        prediction_string = self.index_to_strings(p)
        
        out = {'logits':logits, 'p':p, 'prediction':prediction, 'z':z, 'prediction_string':prediction_string}
        
        return out
    
    def setup_loss(self, features, y, y_):
        
        char_num = self.hparams['char_num']
        x = tf.one_hot(y, char_num)

        logits, p, prediction, xohe = y_['logits'], y_['p'], y_['prediction'], y_['x']

        loss = multiSoftmaxCrossEntropy(xohe, logits)

        accuracy = index_accuracy(prediction, x)
        tf.summary.scalar('accuracy', accuracy)
        
        learning_rate = self.hparams['learning_rate']

        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
        return loss, train_op

    def setup_eval(self, x, labels, y_):

        logits, p, prediction = y_['logits'], y_['p'], y_['prediction']

        accuracy = tf.metrics.accuracy(labels=tf.reshape(labels, (-1, 1)), predictions=tf.reshape(prediction, (-1, 1)))
        
        return {'accuracy' : accuracy}

def string2index(s, MAX_LEN, char_map):
    idx = np.zeros(MAX_LEN, np.uint8)
    for i, c in enumerate(s):
        idx[i] = char_map[c]
    return idx
        
def makeIter(home, epochs, batch_size, MAX_LEN, buffer_size, chunk_size=2**13, test=False):
    
    CMPATH = os.path.join(home, 'char_map.pickle')
    char_map = myPickle.load(CMPATH)
    char_num = len(char_map)
    
    XPATH = os.path.join(home, 'X.h5df') 
        
    if test:
        key = 'test' 
    else:
        key = 'train' 
    
    with h5py.File(XPATH, 'r') as f:
        f = f[key]
        N = len(f)
    
    def G(*args):
        with h5py.File(XPATH, 'r') as f:
            f = f[key]
            bn = math.ceil(N / chunk_size)
            for i in range(bn):
                s = i * chunk_size
                e = (i+1) * chunk_size
                Xchunk = f[s:e]
                for x in Xchunk:
                    yield x
    
    def batch():
        dataset = tf.data.Dataset.from_generator(G, tf.int32, (MAX_LEN,))
        if not test:
            dataset = dataset.repeat(epochs)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        iterator = dataset.make_one_shot_iterator()
        x = iterator.get_next()
        return {'x':x}, x
    
    return batch, char_num, N


def makeValidationIter(home):
    XPATH = os.path.join(home, 'rfX.npy') 
    o = np.load(XPATH)
    o = o.astype(np.int32)
    it = tf.estimator.inputs.numpy_input_fn(o, shuffle=False)()
    r = it[:,0]
    f = it[:,1]
    x = it[:,2:]
    return {'x':x, 'f':f, 'r':r}, x
