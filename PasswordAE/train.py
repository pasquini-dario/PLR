import tensorflow as tf
import numpy as np
import gin
import math
import sys, os
import myPickle

import architecture
import MMAE, AE


model_dir_home = './HOME/MODELS/'
CM = 'charmap.pickle'

@gin.configurable
def setup(model_path, home, reg_type, arch_id, latent_size, epochs, batch_size, max_len, learning_rate, **conf):

    cm_ = os.path.join(home, CM)
    cm = myPickle.load(cm_)
    chars = np.array([x[0] for x in sorted(cm.items(), key=lambda x: x[1])])
    
    if reg_type == 0:
        print("MMAE")
        from AE import makeIter
        train, char_num, M = makeIter(home, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'])
        test, _, _ = makeIter(home, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'], test=True)

    elif reg_type == 1:
        print("Single NoisingAE")
        from denoising import makeIterNoise as makeIter

        train, char_num, M = makeIter(home, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'])
        test, _, _ = makeIter(home, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'], test=True)
    elif reg_type == 2:
        print("MaskedAE")
        from denoising import makeIterMask as makeIter

        mask_size = conf['mask_size']
        train, char_num, M = makeIter(home, mask_size,  epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'])
        test, _, _ = makeIter(home, mask_size, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'], test=True)
    elif reg_type == 3:
        print("NoisingAE")
        from denoising import makeIterMNoise as makeIter
        
        holes_number = conf['holes_number']

        train, char_num, M = makeIter(home, holes_number, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'])
        test, _, _ = makeIter(home, holes_number, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'], test=True)
    elif reg_type == 4:
        print("Single NoisingAE WITH END CHAR")
        from denoising import makeIterNoise as makeIter

        train, char_num, M = makeIter(home, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'], include_end_char=True)
        test, _, _ = makeIter(home, epochs, batch_size, max_len, buffer_size=conf['BUFFER_SIZE'], test=True, include_end_char=True)
    else:
        sys.exit(1)
    
    if arch_id == 0:
        enc = architecture.enc0_16
        dec = architecture.dec0_16
    elif arch_id == 1:
        enc = architecture.enc1_16
        dec = architecture.dec1_16
    elif arch_id == 2:
        enc, dec = architecture.ARCH_resnetBNK0
    elif arch_id == 3:
        enc, dec = architecture.ARCH_resnetBNK1
    elif arch_id == 4:
        enc, dec = architecture.ARCH_resnetBNK2
    elif arch_id == 5:
        enc, dec = architecture.ARCH_resnetBNK3
    elif arch_id == 6:
        enc, dec = architecture.ARCH_resnetBNK4
    elif arch_id == 7:
        enc, dec = architecture.ARCH_INAE
    elif arch_id == 8:
        enc, dec = architecture.ARCH_INAE2
    else:
        print('NO SUCH ARCH_ID')
        sys.exit(1)
    
    N = math.ceil( ( (epochs*M)  / batch_size ) )
    print('Train_iter: ', N)   

    hparams = {
        'enc_arch' : enc,
        'dec_arch' : dec,
        'learning_rate' : learning_rate,
        'char_num' : char_num,
        'batch_size' : batch_size,
        
        'loss_id' : conf.get('loss_id', 0),
        
        'alpha' : conf['alpha'], # scalar loss latent space
        'beta' : conf['beta'], # scalar loss data space
        'latent_size' : latent_size,
        
        'chars' : chars,
    }

    # make estimator
    ae = MMAE.MMAE(model_path, hparams)

    run_conf = ae.setupRunConfig(conf['SAVE_SUMMARY_STEPS'], conf['SAVE_CHECKPOINT_STEP'], keep_checkpoint_max=1)
    estimator = ae(run_conf)
    
    train_spec = tf.estimator.TrainSpec(train, max_steps=N)
    eval_spec = tf.estimator.EvalSpec(test, steps=None, throttle_secs=conf['THROTTLE_SECS'])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    return ae, max_len
    
if __name__ == '__main__':
    
    try:
        gin_conf = sys.argv[1]
    except:
        print("USAGE: CONF")
        sys.exit(1)
        
   
    gin.parse_config_file(gin_conf)
    
    # SETUP MODEL_DIR AND HUB_DIR
    conf_name = os.path.basename(gin_conf).split('.')[0]    
    model_path = os.path.join(model_dir_home, 'CHECKPOINT', conf_name)
    hub_path = os.path.join(model_dir_home, 'HUB', conf_name)
    print(model_path)
    print(hub_path)
    if not ( os.path.join(model_dir_home, 'HUB') and os.path.join(model_dir_home, 'CHECKPOINT')):
        print('NO DIRECTORIES')
        sys.exit(1)
    #######################################
    
    tf.logging.set_verbosity(tf.logging.INFO)    
   
    # TRAIN
    model, max_len = setup(model_path)
    
    # EXPORT
    model.exportHubModule(hub_path, input_shape=(max_len,))
