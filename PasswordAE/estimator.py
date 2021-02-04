import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

TRAIN, PREDICT, EVAL = tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL

def getSessionConfing():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

class Estimator:
    
    NAME = 'Model'
    
    def __init__(self, model_dir, hparams={}, gpuid=-1):
        self.model_dir = model_dir
        self.hparams = hparams
        
    def setup_predictions(self, features, training):
        pass
        
    def setup_loss(self, features, y, y_):
        pass
    
    def setup_eval(self, features, labels, y_):
        pass
        
    def __call__(self, config=None):

        def model_function(features, labels, mode):

            training = (mode == TRAIN)
            
            with tf.variable_scope(self.NAME, reuse=tf.AUTO_REUSE):
                predictions = self.setup_predictions(features, training=training)   
         
            loss, train_op, eval_metric_ops = None, None, None

            if mode == TRAIN:
                loss, train_op = self.setup_loss(features, labels, predictions)

            if mode == EVAL:
                loss, _ = self.setup_loss(features, labels, predictions)
                eval_metric_ops = self.setup_eval(features, labels, predictions)

            return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              loss=loss,
              train_op=train_op,
              eval_metric_ops=eval_metric_ops,
            )

        estimator = tf.estimator.Estimator(
            model_fn=model_function,
            model_dir=self.model_dir,
            config=config
        )
        
        self.estimator = estimator
        return estimator

    @staticmethod
    def setupRunConfig(save_summary_steps, save_checkpoints_steps, keep_checkpoint_max=1):
        session_config = getSessionConfing()
        return tf.estimator.RunConfig(save_summary_steps=save_summary_steps, save_checkpoints_steps=save_checkpoints_steps, keep_checkpoint_max=keep_checkpoint_max, session_config=session_config)
    
    def exportHubModule(self, path, input_shape, trainable=False):

        self.input_shape = input_shape

        spec = hub.create_module_spec(self.hub_model_fn)
        module = hub.Module(spec, name=Estimator.NAME, trainable=trainable)
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.estimator.latest_checkpoint())
            sess.run(tf.tables_initializer())
            module.export(path, sess)
