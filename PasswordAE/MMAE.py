from AE import *
from MMD import mmd_loss

def multiSigmoidEntropy(label, logits):
    num_classes = logits.shape.as_list()[-1]
    assert num_classes == label.shape.as_list()[-1]
    logits = tf.reshape(logits, [-1, num_classes])
    label = tf.reshape(label, [-1, num_classes])
    return tf.losses.sigmoid_cross_entropy(label, logits)

def index_accuracy(a, b):
    eq = tf.cast(tf.equal(a, b), tf.float32)
    acc = tf.reduce_mean( eq )
    return acc

class MMAE(AE):
    
    def getTargetLatentSample(self, shape):
        return tf.random_normal(shape)
    
    
    def setup_predictions_SampleFromLatent(self, x, stddev, n, training):

        char_num = self.hparams['char_num']
        enc_arch = self.hparams['enc_arch']
        dec_arch = self.hparams['dec_arch']
        latent_size = self.hparams['latent_size']    
        
        ## CMAP
        latent_size = self.hparams['latent_size'] 
        chars = self.hparams['chars']
        self.char_map = tf.contrib.lookup.index_to_string_table_from_tensor(chars)
        ##
        
        x = tf.one_hot(x, char_num)

        with tf.variable_scope(ENC, reuse=tf.AUTO_REUSE) as scope:
            z = enc_arch(x, latent_size, training)
        
        z_ = self.sampleLatent(z, stddev, n)

        out_shape = x.shape.as_list()[1:]
        with tf.variable_scope(DEC, reuse=tf.AUTO_REUSE) as scope:
            logits = dec_arch(z_, out_shape, training)

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
        xohe = tf.one_hot(y, char_num)

        logits, p, prediction, z = y_['logits'], y_['p'], y_['prediction'], y_['z']
        alpha = self.hparams['alpha']
        beta = self.hparams['beta']
        learning_rate = self.hparams['learning_rate']
        batch_size = self.hparams['batch_size']

        loss_id = self.hparams['loss_id']
        
        if loss_id == 0:
            print("Softmax loss")
            rec_err = multiSoftmaxCrossEntropy(xohe, logits)
        elif loss_id == 1:
            print("Sigmoid loss")
            rec_err = multiSigmoidEntropy(xohe, logits)
        else:
            raise Execption("No such loss")

        rec_err = rec_err * beta
        
        tf.summary.scalar('rec_err', rec_err)

        if alpha:
            shape = batch_size, z.shape.as_list()[1]
            ztarget = self.getTargetLatentSample(shape)
            latent_reg = mmd_loss(z, ztarget) * alpha
            tf.summary.scalar('latent_reg', latent_reg)    
            loss = rec_err + latent_reg
        else:
            print("NO REG ON LATENT")
            loss = rec_err 


        accuracy = index_accuracy(prediction, y)
        tf.summary.scalar('accuracy', accuracy)

        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
        return loss, train_op

    def sampleLatent(self, mean, stddev, n):
        m = mean.shape.as_list()[-1]
        z = tf.random_normal((n[0], m), mean, stddev)
        return z


    def hub_model_fn(self):

        training = False
        alpha = self.hparams['alpha']

        xph = tf.placeholder(tf.int32, shape=(None, *self.input_shape))
        
        features = {'x':xph}
        out = self.setup_predictions(features, training)
        out.update({'default':out['p']})
        #out.pop('prediction_string')

        hub.add_signature(inputs=xph, outputs=out)

        if True:
            ###################
            if alpha:
                ls = self.hparams['latent_size']

                zph = tf.placeholder(tf.float32, shape=(None, ls))

                out_shape = out['x'].shape.as_list()[1:]
                out = self.setup_predictions_latent(zph, out_shape, training)
                out.update({'default':out['p']})
                hub.add_signature(name='latent', inputs=zph, outputs=out)
            ####################

            xph = tf.placeholder(tf.int32, shape=(1, *self.input_shape))
            stddevph = tf.placeholder(tf.float32, shape=(1,))
            nph = tf.placeholder(tf.int32, shape=1)

            out = self.setup_predictions_SampleFromLatent(xph, stddevph, nph, training)
            out.update({'default':out['p']})

            inputs = {
                    'x' : xph,
                    'stddev' : stddevph,
                    'n' : nph,
                    }

            hub.add_signature(name='sample_from_latent', inputs=inputs, outputs=out)

