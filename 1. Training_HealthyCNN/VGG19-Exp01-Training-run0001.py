#!/usr/bin/env python
# coding: utf-8

# # Transfer learning of a VGG19 network to perform classification on the the CIFAR 100 dataset
# Perform transfer learning on a VGG19 network that was previously trained on the ImageNet dataset to perform classification on the CIFAR100 dataset using 3 rounds of training
# - On round 1, unfreeze and train all weights
# - On round 2, freeze the first 3 convolutional blocks, train the last 2 convolutional blocks and dense layers
# - On round 3, fine-tune on the dense layers only

# # Input Parameters
# Modifiable inputs to the training task

# In[ ]:


# Path to locally stored pre-trained weights that are available from Tensorflow 2.0
pretrained_path = '../Pretrained_Weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


# In[ ]:


# Batch size
BATCH_SIZE=128


# In[ ]:


# Number of epochs in each training round
EPOCHS1=20
EPOCHS2=10
EPOCHS3=10


# In[ ]:


# Learning rate in each training round
LR1=3e-5
LR2=3e-5
LR3=3e-5


# In[ ]:


# Dropout rate
dropout = 0.3


# # Output Filename and path

# In[ ]:


# Modifiable value of the run ID
run = '0001'


# In[ ]:


# Path to directory storing the trained model
trained_dir = '../Trained_Models'
trained_prefix = '/VGG19-Exp01-TrainedModel-run'
trained_suffix = '.h5'


# In[ ]:


# Filename of trained model
output_filename = trained_dir + trained_prefix + str(run) + trained_suffix
print("Will save to:\n\t",output_filename)


# # Load modules

# In[ ]:


import tensorflow.keras as K


# # Import Data

# In[ ]:


def preprocess_data(X, Y):
    """Pre-processes the data for the trained model"""
    # Input is pre-processed input to match VGG19 using built in function
    X_p = K.applications.vgg19.preprocess_input(X.astype('float32')) 
    
    # One-hot matrix
    Y_p = K.utils.to_categorical(Y, classes)
    
    return X_p, Y_p


# In[ ]:


# Load the raw cifar100 dataset
(raw_X_train, raw_Y_train), (raw_X_valid, raw_Y_valid) = K.datasets.cifar100.load_data()
classes = 100


# In[ ]:


# Pre-process the training and validation data
X_train, Y_train = preprocess_data(raw_X_train, raw_Y_train)  # training
X_valid, Y_valid = preprocess_data(raw_X_valid, raw_Y_valid)  # validation


# # Set up the model

# ## Pre-trained VGG19's convolutional blocks

# In[ ]:


# Initialize the convolutional blocks from TF's built-in function
vgg = K.applications.VGG19(include_top=False, 
                           weights=None, 
                           pooling='avg'
                          )

# Load the pre-trained weights
vgg.load_weights(pretrained_path)

# Set-up VGG19's convolutional blocks with an input and output
output = vgg.layers[-1].output
output = K.layers.Flatten()(output)
vgg_model = K.Model(vgg.input, output, name="VGG19_ConvBlocks")
vgg_model.trainable = True


# In[ ]:


# Display the convolutional block model summary
vgg_model.summary()


# ## VGG19 with new dense and classification layers

# In[ ]:


# Build sequential model
model = K.Sequential()

# INPUT
# Up sample
model.add(K.Input(shape=(32,32,3)))
model.add(K.layers.UpSampling2D(4))

# VGG convolutional blocks
model.add(vgg_model)

# First hidden dense layer after VGG
model.add(K.layers.Dense(1000, activation='relu'))
model.add(K.layers.Dropout(dropout))

# Second hidden dense layer after VGG
model.add(K.layers.Dense(1000, activation='relu'))
model.add(K.layers.Dropout(dropout))

# Classification layer
model.add(K.layers.Dense(classes, activation='softmax'))

# Compile
model.compile(optimizer=K.optimizers.RMSprop(lr=LR1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Retrieve weights before training
weights_before = model.get_weights()


# In[ ]:


# Display the model summary
# See above for the details of the "VGG19_ConvBlocks"
model.summary()


# # Train the model

# ## First Training Round
# - Train all layers

# In[ ]:


# Unfreeze all layers
set_trainable = True
for layer in vgg_model.layers:
    if set_trainable:
        layer.trainable = True


# In[ ]:


# Recompile model
model.compile(optimizer=K.optimizers.RMSprop(lr=LR1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Print out what layers of the convolutional blocks are being trained and what layers are held fixed
for layer in vgg_model.layers:
    print("Layer", layer.name, " :\t", layer.trainable)


# In[ ]:


# Print out what layers of the overall model are being trained and what layers are held fixed
for layer in model.layers:
    print("Layer", layer.name, " :\t", layer.trainable)


# In[ ]:


# Run training
# First round of training: Train all layers
print('\n\n FIRST ROUND\n')
history = model.fit(x=X_train, y=Y_train,
                    validation_data = (X_valid, Y_valid),
                    epochs=EPOCHS1,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_steps=10
                   )


# ## Second Round
# - Freeze convolutional blocks 1-3
# - Train convolutional blocks 4-5 and dense layers

# In[ ]:


# Freeze convolutional blocks 1-3 in VGG19
# Leave convolutional blocks 4-5 as trainable
set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True

    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# In[ ]:


# Recompile model
model.compile(optimizer=K.optimizers.RMSprop(lr=LR2),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Print out what layers of the convolutional blocks are being trained and what layers are held fixed
for layer in vgg_model.layers:
    print("Layer", layer.name, " :\t", layer.trainable)


# In[ ]:


# Print out what layers of the overall model are being trained and what layers are held fixed
for layer in model.layers:
    print("Layer", layer.name, " :\t", layer.trainable)


# In[ ]:


# Run training
# Second round of training: Train conv.blocks 4-5 and dense layers
print('\n\n SECOND ROUND\n')
history = model.fit(x=X_train, y=Y_train,
                    validation_data = (X_valid, Y_valid),
                    epochs=EPOCHS2,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_steps=10
                   )


# ## Third Round
# - Freeze all convolutional blocks
# - Train dense layers

# In[ ]:


# Freeze all VGG layers
set_trainable = False
for layer in vgg_model.layers:
    if set_trainable == True:
        layer.trainable = True
    else:
        layer.trainable = False


# In[ ]:


# Recompile model
model.compile(optimizer=K.optimizers.RMSprop(lr=LR3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Print out what layers of the convolutional blocks are being trained and what layers are held fixed
for layer in vgg_model.layers:
    print("Layer", layer.name, " :\t", layer.trainable)


# In[ ]:


# Print out what layers of the overall model are being trained and what layers are held fixed
for layer in model.layers:
    print("Layer", layer.name, " :\t", layer.trainable)


# In[ ]:


# Run training
# Third round of training: Train dense layers only
print('\n\n THIRD ROUND\n')
history = model.fit(x=X_train, y=Y_train,
                    validation_data = (X_valid, Y_valid),
                    epochs=EPOCHS3,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_steps=10
                   )


# # Save Model

# In[ ]:


model.save(output_filename)


# # Run evaluation on the final model

# In[ ]:


score = model.evaluate(X_valid, Y_valid, verbose=0)


# In[ ]:


print("\n")
print('='*20)
print("\n")
print(output_filename)
print("Validation accuracy:", score[1])

