# Springboard_Capstone_2_Final

## Background: 
In this competition, we were challenged to use the Speech Commands Dataset to build an algorithm that understands
simple spoken commands.By improving the recognition accuracy of open-sourced voice interface tools,
we can improve product effectiveness and their accessibility.

The <b>training set </b> provided contains a few informational files and a folder of audio files. 
The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. 
There are more labels that should be predicted. The labels that we need to predict in Test are yes, no, up, down, left, right, on, off, 
stop, go. Everything else should be considered either unknown or silence. 
The folder _background_noise_ contains longer clips of "silence" that we can break up and use as training input.

The <b>test set </b> contains an audio folder with 150,000+ files in the format clip_000044442.wav. 
The task is to predict the correct label.

## Pre-processing: 
Exploratory data analysis showed that spectrogram representations of audio files could serve as better input to a deep learning network 
compared to raw audio inputs or fourier representations of the audio files. Further exploration showed that Mel power spectrograms 
, which better mimic the way the human ear perceives sound, could potentially be better than log power spectrograms. MFCC's which further abstract
away information from the mel spectrogram can also be potentially used. <br>
I generated log spectrograms, mel spectrograms and mfccs and stored them in the form of numpy arrays. I had previously tried to convert log spectrograms
into color images and store them to be used as input. But I found storing them as numpy arrays to be more memory efficient and it yielded 
the same results. <br>
So ultimately I had three numpy arrays 1) Log Spec 2) Mel Spec 3) MFCC corresponding to different representations of the input 

~~~
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

def mel_specgram(samples, sample_rate):
    
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def mfcc(samples,sample_rate):
    S_spec = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    log_S= librosa.power_to_db(S_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return mfcc
 ~~~ 
![Image of Mel Spec](https://github.com/ravimaranganti/Springboard_Capstone_2_Final/blob/master/images/melspec_yes.png)
 <p align="center"> <b> Mel Spectrograms of Randomly Selected audiofiles labeled 'yes' </b> </p>
<br>
<br>

## Models: 
### Light-CNN: 
I started out with a very simple CNN architecture and used log spectrograms as input. 
~~~
input_shape = (99, 81, 1)
nclass = 12
inp = Input(shape=input_shape)
norm_inp = BatchNormalization()(inp)
img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(norm_inp)
img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Flatten()(img_1)

dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam()
callbacks = [EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')]
model.compile(optimizer=opt, loss=losses.binary_crossentropy)
model.summary()

#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2017)
history=model.fit_generator(generator=generator(train_file_map,batch_size=250, mode='train'),steps_per_epoch=int(np.ceil(len(sample_map_df_train)/250)), validation_data=generator(train_file_map,batch_size=250, mode='val'), validation_steps=int(np.ceil(len(sample_map_df_val)/250)), epochs=3, shuffle=True, callbacks=callbacks)

model.save(os.path.join(model_path, 'light_cnn.model'))
~~~
During training, I achieved high accuracies on the validation set using this light cnn model but this accuracy did not translate to good performance on the hold out test set at kaggle. When I read the forums to understand why this may be, it was suggested that the test set may be from a different distribution especially when it came to unknown uknowns. For instance, here is the confusion matrix on the validation set which I got during training. While accuracies on training and validation sets were close to 96%, the accuracy I achieved on the test set was only about 74%. 
![Image of Confusion Matrix](https://github.com/ravimaranganti/Springboard_Capstone_2_Final/blob/master/images/confusion_matrix_validation_lightcnn.png)
<p align="center"> <b> Confusion Matrix on Validation Set: Light CNN </b> </p>
<br>

### Resnet: 
Since this simple CNN architecture was not generalizing well on the test set, I decided to use a Resnet style architecture. Resnets use residual blocks and the presence of residual blocks can help build deeper networks. 

~~~
class ResNet():
    """
    Usage: 
        sr = ResNet([4,8,16], input_size=(50,50,1), output_size=12)
        sr.build()
        followed by sr.m.compile(loss='categorical_crossentropy', 
                                 optimizer='adadelta', metrics=["accuracy"])
        save plotted model with: 
            keras.utils.plot_model(sr.m, to_file = '<location>.png', 
                                   show_shapes=True)
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNet
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        i = Input(shape = self.input_size, name = 'input')
        x = Conv2D(self.filters_list[0], (3,3), 
                   padding = 'same', 
                   kernel_initializer = self.initializer)(i)
        x = MaxPooling2D(padding = 'same')(x)        
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        if len(self.filters_list) > 1:
            for filt in self.filters_list[1:]:
                x = Conv2D(filt, (3,3),
                           strides = (2,2),
                           padding = 'same',
                           activation = 'relu',
                           kernel_initializer = self.initializer)(x)
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.output_size, activation = 'softmax')(x)
        
        self.m = Model(i,x)
        return self.m
   ~~~
<br>
<br>
