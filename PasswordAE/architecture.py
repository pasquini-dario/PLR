import tensorflow as tf

FILTER_SIZE = 3

def enc0_16(x, latent_size, training):
    x = tf.layers.conv1d(x, 64, FILTER_SIZE, strides=1, padding='same', activation=tf.nn.relu)
    x = tf.layers.conv1d(x, 128, FILTER_SIZE, strides=2, padding='same', activation=tf.nn.relu)
    x = tf.layers.conv1d(x, 128, FILTER_SIZE, strides=1, padding='same', activation=tf.nn.relu)
    x = tf.layers.conv1d(x, 128, FILTER_SIZE, strides=1, padding='same', activation=tf.nn.relu)
    x = tf.layers.conv1d(x, 256, FILTER_SIZE, strides=2, padding='same', activation=tf.nn.relu)
    x = tf.layers.flatten(x) 
    x = tf.layers.dense(x, latent_size, activation=None)
    return x

def dec0_16(x, output_shape, training):
    output_size = output_shape[0] * output_shape[1]
    x = tf.layers.dense(x, 4 * 256, activation=tf.nn.relu)
    x = tf.reshape(x, (-1, 4, 1, 256))
    x = tf.layers.conv2d_transpose(x, 256, FILTER_SIZE, strides=(2, 1), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, 128, FILTER_SIZE, strides=(1, 1), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, 128, FILTER_SIZE, strides=(1, 1), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, 128, FILTER_SIZE, strides=(2, 1), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, 64, FILTER_SIZE, strides=(1, 1), padding='same', activation=tf.nn.relu)
    x = tf.reshape(x, (-1, 16, 64))
    x = tf.layers.conv1d(x, output_shape[1], FILTER_SIZE, strides=1, padding='same')
    return x

ARCH0_16 = [enc0_16, dec0_16]

def enc1_16(x, latent_size, training):
    x = tf.layers.conv1d(x, 64, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
    print(x.shape)
    x = tf.layers.conv1d(x, 128, kernel_size=3, strides=2, padding='same',  activation=tf.nn.relu)
    print(x.shape)
    x = tf.layers.conv1d(x, 256, kernel_size=3, strides=2, padding='same',  activation=tf.nn.relu)
    print(x.shape)
    x = tf.layers.flatten(x)
    print(x.shape)
    x = tf.layers.dense(x, latent_size, use_bias=False)
    return x

def dec1_16(x, output_shape, training):
    output_size = output_shape[0] * output_shape[1]
    x = tf.layers.dense(x, 4 * 256, activation=tf.nn.relu)
    x = tf.reshape(x, (-1, 4, 1, 256))
    print(x.shape)
    x = tf.layers.conv2d_transpose(x, 256, 3, strides=(2, 1), padding='same', activation=tf.nn.relu)
    print(x.shape)
    x = tf.layers.conv2d_transpose(x, 128, 3, strides=(2, 1), padding='same', activation=tf.nn.relu)
    print(x.shape)
    x = tf.reshape(x, (-1, 16, 128))
    x = tf.layers.conv1d(x, output_shape[1], 5, strides=1, padding='same')
    print(x.shape)
    return x

ARCH1_16 = [enc1_16, dec1_16]

########################################################################################

def ResBlockDeepBNK(inputs, dim, with_batch_norm=True, training=True):
    x = inputs

    dim_BNK = dim // 2

    if with_batch_norm:
        x = tf.layers.batch_normalization(x, training=training)

    x = tf.nn.relu(x)
    x = tf.layers.conv1d(x, dim_BNK, 1, padding='same')

    if with_batch_norm:
        x = tf.layers.batch_normalization(x, training=training)

    x = tf.nn.relu(x)
    x = tf.layers.conv1d(x, dim_BNK, 5, padding='same')

    if with_batch_norm:
        x = tf.layers.batch_normalization(x, training=training)

    x = tf.nn.relu(x)
    x = tf.layers.conv1d(x, dim, 1, padding='same')

    return inputs + (0.3*x)



def encResnetBNK(x, latent_size, training=False):
    batch_norm = False
    layer_dim = 128
    x = tf.layers.conv1d(x, layer_dim, 5, padding='same')
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    #x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    #print(x.shape)
    x = tf.layers.flatten(x)
    print(x.shape)
    logits = tf.layers.dense(x, latent_size)
    return logits

def decResnetBNK(x, output_shape, training=False):
    batch_norm = False
    layer_dim = 128

    x = tf.layers.dense(x, output_shape[0] * layer_dim)
    print(x.shape)
    x = tf.reshape(x, [-1, output_shape[0], layer_dim])
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    #x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    #print(x.shape)
    logits = tf.layers.conv1d(x, output_shape[1], 1, padding='same')
    print(logits.shape)
    return logits

ARCH_resnetBNK0 = [encResnetBNK, decResnetBNK]

########################################################################################

def encResnetBNK1(x, latent_size, training=False):
    batch_norm = False
    layer_dim = 128
    x = tf.layers.conv1d(x, layer_dim, 5, padding='same')
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = tf.layers.flatten(x)
    print(x.shape)
    logits = tf.layers.dense(x, latent_size)
    return logits

def decResnetBNK1(x, output_shape, training=False):
    batch_norm = False
    layer_dim = 128

    x = tf.layers.dense(x, output_shape[0] * layer_dim)
    print(x.shape)
    x = tf.reshape(x, [-1, output_shape[0], layer_dim])
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    logits = tf.layers.conv1d(x, output_shape[1], 1, padding='same')
    print(logits.shape)
    return logits

ARCH_resnetBNK1 = [encResnetBNK1, decResnetBNK1]


########################################################################################

def encResnetBNK2(x, latent_size, training=False):
    batch_norm = False
    layer_dim = 128
    x = tf.layers.conv1d(x, layer_dim, 5, padding='same')
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = tf.layers.flatten(x)
    print(x.shape)
    logits = tf.layers.dense(x, latent_size)
    return logits

def decResnetBNK2(x, output_shape, training=False):
    batch_norm = False
    layer_dim = 128

    x = tf.layers.dense(x, output_shape[0] * layer_dim)
    print(x.shape)
    x = tf.reshape(x, [-1, output_shape[0], layer_dim])
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    logits = tf.layers.conv1d(x, output_shape[1], 1, padding='same')
    print(logits.shape)
    return logits

ARCH_resnetBNK2 = [encResnetBNK2, decResnetBNK2]

########################################################################################


def encResnetBNK3(x, latent_size, training):
    batch_norm = True
    layer_dim = 128
    x = tf.layers.conv1d(x, layer_dim, 5, padding='same')
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = tf.layers.flatten(x)
    print(x.shape)
    logits = tf.layers.dense(x, latent_size)
    return logits

def decResnetBNK3(x, output_shape, training):
    batch_norm = True
    layer_dim = 128

    x = tf.layers.dense(x, output_shape[0] * layer_dim)
    print(x.shape)
    x = tf.reshape(x, [-1, output_shape[0], layer_dim])
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    logits = tf.layers.conv1d(x, output_shape[1], 1, padding='same')
    print(logits.shape)
    return logits

ARCH_resnetBNK3 = [encResnetBNK3, decResnetBNK3]

########################################################################################

def encResnetBNK4(x, latent_size, training=False):
    batch_norm = False
    layer_dim = 128
    x = tf.layers.conv1d(x, layer_dim, 5, padding='same')
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = tf.layers.flatten(x)
    print(x.shape)
    logits = tf.layers.dense(x, latent_size)
    return logits

def decResnetBNK4(x, output_shape, training=False):
    batch_norm = False
    layer_dim = 128

    x = tf.layers.dense(x, output_shape[0] * layer_dim)
    print(x.shape)
    x = tf.reshape(x, [-1, output_shape[0], layer_dim])
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    logits = tf.layers.conv1d(x, output_shape[1], 1, padding='same')
    print(logits.shape)
    return logits

ARCH_resnetBNK4 = [encResnetBNK4, decResnetBNK4]

########################################################################################

def INAE_enc(x, latent_size, training=False):
    batch_norm = False
    layer_dim = 128
    x = tf.layers.conv1d(x, layer_dim, 5, padding='same')
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    return x

def INAE_dec(x, output_shape, training=False):
    batch_norm = False
    layer_dim = 128

    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    logits = tf.layers.conv1d(x, output_shape[1], 1, padding='same')
    print(logits.shape)
    return logits

ARCH_INAE = [INAE_enc, INAE_dec]

########################################################################################

def INAE_enc1(x, latent_size, training=False):
    batch_norm = False
    layer_dim = 128
    x = tf.layers.conv1d(x, layer_dim, 5, padding='same')
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    return x

def INAE_dec1(x, output_shape, training=False):
    batch_norm = False
    layer_dim = 128
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=training)
    print(x.shape)
    logits = tf.layers.conv1d(x, output_shape[1], 1, padding='same')
    print(logits.shape)
    return logits

ARCH_INAE2 = [INAE_enc1, INAE_dec1]

########################################################################################

