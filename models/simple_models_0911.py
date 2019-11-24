from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
                          
def deep(features_shape, num_classes, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Flatten
    o = Flatten(name='flatten')(o)
    
    # Dense layer
    o = Dense(512, activation=act, name='dense1')(o)
    o = Dense(512, activation=act, name='dense2')(o)
    o = Dense(512, activation=act, name='dense3')(o)
    
    # Predictions
    o = Dense(num_classes, activation='softmax', name='pred')(o)
    
    # Print network summary
    Model(inputs=x, outputs=o).summary()
    
    return Model(inputs=x, outputs=o)

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D

def deep_cnn(features_shape, num_classes, act='relu'):

    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Block 1
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block1_conv', input_shape=features_shape)(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)
    
    # Block 2
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)

    # Block 3
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block3_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)

    # Flatten
    o = Flatten(name='flatten')(o)
    
    # Dense layer
    o = Dense(64, activation=act, name='dense')(o)
    o = BatchNormalization(name='dense_norm')(o)
    o = Dropout(0.2, name='dropout')(o)
    
    # Predictions
    o = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=o).summary()
    
    return Model(inputs=x, outputs=o)

def light_cnn(features_shape, num_classes, act='relu'):
    inp = Input(shape=features_shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Conv2D(16, (2,2), activation=act)(norm_inp)
    img_1 = Conv2D(16, (2,2), activation=act)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Conv2D(32, (3,3), activation=act)(img_1)
    img_1 = Conv2D(32, (3,3), activation=act)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Conv2D(64, (3,3), activation=act)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=act)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=act)(dense_1))
    dense_1 = Dense(num_classes, activation='softmax')(dense_1)
    
    Model(inputs=inp,outputs=dense_1).summary()
    return Model(inputs=inp,outputs=dense_1)

def deep_cnn2(features_shape, num_classes, act='relu'):
    inp = Input(shape=features_shape)
    norm_inp = BatchNormalization()(inp)
    img_1= Conv2D(24, (4,4), activation=act)(norm_inp)
    img_1= MaxPooling2D(pool_size=(2,2))(img_1)
    img_1= Dropout(rate=0.25)(img_1)
    img_1= Conv2D(48, (3,3), activation=act)(img_1)
    img_1= ZeroPadding2D(((0,0),(0,1)))(img_1)
    img_1 = MaxPooling2D(pool_size=(2,2))(img_1)
    img_1 = Dropout(rate=0.25)(img_1)
    img_1 = Conv2D(96, (4,4), activation=act)(img_1)
    img_1 = MaxPooling2D(pool_size=(2,2))(img_1)
    img_1 = Dropout(rate=0.25)(img_1)
    img_1 = Flatten()(img_1)
    
    dense_1 = BatchNormalization()(Dense(256, activation=act)(img_1))
   # dense_1 = BatchNormalization()(Dense(256, activation=act)(dense_1))
    dense_1 = Dense(num_classes, activation='softmax')(dense_1)
    
    Model(inputs=inp,outputs=dense_1).summary()
    return Model(inputs=inp,outputs=dense_1)
