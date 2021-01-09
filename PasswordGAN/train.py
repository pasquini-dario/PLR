from PasswordGAN.FAIL import rawModel

from PasswordGAN import architectures
import tensorflow_hub as hub
import pickle
import numpy as np
import gin
import os, sys

from PasswordGAN.vanilla import Vanilla

HOME = './HOME/'
rawModel.CHECK_ROOT_DIR = os.path.join(HOME, 'CHECKPOINTS/')

def arch_by_id(arch_id):
    return architectures.archs[arch_id]

@gin.configurable
def setup(
    arch_id,
    dataset_path,
    z_size,
    z_prior,
    x_size,
    learning_rate,
    beta1,
    beta2,
    D_iters,
    batch_size,
    epochs,
    # evaluation
    evaluation_fq,
    generator_samples,
    evaluation_batch_size,
    
    gamma=0,
    alpha=0,
    
    PassGAN_type=0
):
    
    arch = arch_by_id(arch_id)  
    
    print(arch)
    
    # LOAD DATASET
    with open(dataset_path, 'rb') as f:
        char_map, (X_train, X_test, X_test_clean) = pickle.load(f)  
            
    params = dict(
        G_maker=arch[0],
        D_maker=arch[1],
        chars=char_map,
        z_size=z_size,
        z_prior=z_prior,
        x_size=x_size,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        D_iters=D_iters,
        
        gamma=gamma,
        alpha=alpha,

        batch_size=batch_size,
        epochs=epochs,

        evaluation=dict(
            evaluation_fq=evaluation_fq,
            test_set=X_test,
            test_set_clean=X_test_clean,
            generator_samples=generator_samples,
            evaluation_batch_size=evaluation_batch_size,
        )
    )

    # PassGAN_type specific
    if PassGAN_type == 0:
        # vanilla and label smoothing
        PassGAN_class = Vanilla
        data = {'x':X_train}
    else:
        sys.exit(1)
    
    return PassGAN_class, params, data


def main():
    global OP_TYPE
    
    try:
        gin_conf = sys.argv[1]
        gin.parse_config_file(gin_conf)

        try:
            # 0 == TRAIN; 1 == EXPORT
            OP_TYPE = int(sys.argv[2])
        except:
            OP_TYPE = 0
    
        try:
            # additionaly name
            name_ext = sys.argv[3]
        except:
            name_ext = ''   
    except:
        print('USAGE: CONF OP_TYPE? NAME_MODIFIER?')
        sys.exit(1)

    conf_name = os.path.basename(gin_conf)
    name = conf_name + name_ext
    
    PassGAN_class, params, data = setup()

    pg = PassGAN_class(name, params=params, learning_rate=params['learning_rate'])

    epochs = params['epochs']
    batch_size = params['batch_size']

    if OP_TYPE == 0:
        pg.train(data, epochs, batch_size, summary_steps_number=100, use_tqdm=True)
    elif OP_TYPE:
        print("EXPORT:\n")
        path = os.path.join(HOME, 'HUBS', pg.name)
        pg.exportHubModule(path , pg.hub_function)

if __name__ == '__main__':
    main()
