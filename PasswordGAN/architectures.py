import tensorflow as tf

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

#################################################################################################################
########### DEEP RESBLOCK BOTTLENECK

def ResBlockDeepBNK(inputs, dim, with_batch_norm=False, training=True):
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

def G0(x, seq_len, output_dim, layer_dim=128, is_training=False, with_logits=False):    
    
    batch_norm = True
    
    x = tf.layers.dense(x, layer_dim * seq_len)
    x = tf.reshape(x, [-1, seq_len, layer_dim])
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    logits = tf.layers.conv1d(x, output_dim, 1, padding='same')
    x = softmax(logits, output_dim)
    if with_logits:
        return x, logits
    return x

def D0(x, seq_len, layer_dim=128, is_training=False):
    x = tf.layers.conv1d(x, layer_dim, 1, padding='same')
    x = ResBlockDeepBNK(x, layer_dim)
    x = ResBlockDeepBNK(x, layer_dim)
    x = ResBlockDeepBNK(x, layer_dim)
    x = ResBlockDeepBNK(x, layer_dim)
    x = ResBlockDeepBNK(x, layer_dim)
    x = tf.reshape(x, [-1, seq_len * layer_dim])
    logits = tf.layers.dense(x, 1)
    return logits

#################################################################################################################

def ResBlockDeepBNK_FS3(inputs, dim, with_batch_norm=False, training=True):
    x = inputs
    
    dim_BNK = dim // 2
    
    if with_batch_norm:
        x = tf.layers.batch_normalization(x, training=training)
        
    x = tf.nn.relu(x)
    x = tf.layers.conv1d(x, dim_BNK, 3, padding='same')
    
    if with_batch_norm:
        x = tf.layers.batch_normalization(x, training=training)
        
    x = tf.nn.relu(x)
    x = tf.layers.conv1d(x, dim_BNK, 5, padding='same')
    
    if with_batch_norm:
        x = tf.layers.batch_normalization(x, training=training)
        
    x = tf.nn.relu(x)
    x = tf.layers.conv1d(x, dim, 3, padding='same')
    
    return inputs + (0.3*x)

def G1(x, seq_len, output_dim, layer_dim=128, is_training=False, with_logits=False):    
    
    batch_norm = True
    
    x = tf.layers.dense(x, layer_dim * seq_len)
    x = tf.reshape(x, [-1, seq_len, layer_dim])
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    logits = tf.layers.conv1d(x, output_dim, 1, padding='same')
    x = softmax(logits, output_dim)
    if with_logits:
        return x, logits
    return x

def D1(x, seq_len, layer_dim=128, is_training=False):
    x = tf.layers.conv1d(x, layer_dim, 1, padding='same')
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = tf.reshape(x, [-1, seq_len * layer_dim])
    logits = tf.layers.dense(x, 1)
    return logits

#################################################################################################################

def G2(x, seq_len, output_dim, layer_dim=128, is_training=False, with_logits=False):    
    
    batch_norm = True
    
    x = tf.layers.dense(x, layer_dim * seq_len, activation=tf.nn.leaky_relu)
    x = tf.layers.dense(x, layer_dim * seq_len)
    x = tf.reshape(x, [-1, seq_len, layer_dim])
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    x = ResBlockDeepBNK_FS3(x, layer_dim, with_batch_norm=batch_norm, training=is_training)
    logits = tf.layers.conv1d(x, output_dim, 1, padding='same')
    x = softmax(logits, output_dim)
    if with_logits:
        return x, logits
    return x

def D2(x, seq_len, layer_dim=128, is_training=False):
    x = tf.layers.conv1d(x, layer_dim, 1, padding='same')
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = ResBlockDeepBNK_FS3(x, layer_dim)
    x = tf.reshape(x, [-1, seq_len * layer_dim])
    logits = tf.layers.dense(x, 1)
    return logits

#################################################################################################################

archs = {
    0 : (G0, D0), #'deep_legacy'
    1 : (G1, D1), #'deep_legacy with kernel 3 rather than 1'
    2 : (G2, D2), #'1 but with two fully connected as input for the generator'
    
    6 : (G0, D0), #'retro-compatibility'
}
