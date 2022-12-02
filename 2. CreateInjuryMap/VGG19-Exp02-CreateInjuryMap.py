#!/usr/bin/env python
# coding: utf-8

# # Creating an Injury Map for the trained VGG19
# 
# Create a "map" of network weight ("synapse") injury in the trained VGG19. 
# 
# The map defines a random order by which to injure the weights, so that injury is random throughout. There are two main purposes of doing this:
# 1. To simulate a diffuse and random "synaptic" injury in the network
# 2. To create a reproducible "disease progression" within a given network
# 
# This allows the experimenter to injure a given network cumulatively so as to simulate progressive neurodegeneration.
# - e.g. When going from a 10% injury to a 15% injury, the script will essentially injure an additional 5% of the network weights on-top of the 10% of weights that were already injured
# 
# **The _output_ is a list of the same length as the _weights_ returned by the tensorflow function model.get_weights()**
# - The elements in the _output_ list correspond to the matching elements the _weights_ list
#     - e.g. list element 0 is for the weights in the first layer, list element 1 is for the weights in the second layer
# - Each list element in the _output_ is a numpy array of the same size as the matching element in the _weights_ list
# - The elements in the _output_ corresponding to the weights (e.g. list element 0, 2, 4, etc for VGG19) are a randomized list of integers that give the order by which to injure (ablate) the weights in that given layer
# - The elements in the _output_ corresponding to the biases (e.g. list elements 1, 3, 5, etc for VGG19) are not randomized, as the biases were not modified in this paper

# # Input: Model source filename

# In[ ]:


# run ID corresponding to the trained model
run = '0001'


# In[ ]:


# Path to directory storing the trained model
trained_dir = '../Trained_Models'
trained_prefix = '/VGG19-Exp01-TrainedModel-run'
trained_suffix = '.h5'


# In[ ]:


# Filename of model to use
model_filename = trained_dir + trained_prefix + str(run) + trained_suffix
print("Will use model:\n\t",model_filename)


# # Output (Injury map) filename

# In[ ]:


# Path to 
map_dir = '../Injury_Maps'
map_prefix = '/VGG19-Exp02-InjuryMap-run'
map_suffix = '.pickle'


# In[ ]:


# Filename
injury_map_path = map_dir + map_prefix + str(run) + map_suffix
print("Injury map path:\n\t",injury_map_path)


# # Load modules

# In[ ]:


import tensorflow.keras as K
import numpy as np
import os
import pickle


# # Load trained model

# In[ ]:


model = K.models.load_model(model_filename)


# In[ ]:


model.summary()


# ## Get weights of the trained model

# In[ ]:


weights = model.get_weights()


# # Define functions

# ## Function to Read variables in and out

# In[ ]:


def pickle_out_variable(variable, filename, folder_path='.'):
    """
    Simple function to pickle out a single python variable
    given the filename (filename.pickle), as a string
    and a foldername ('.' for current directory) as a string
    """
    # Open the file, call it "results.pickle"
    pickle_out = open(os.path.join(folder_path, filename), 'wb')
    # Write out the results to defined file
    pickle.dump(variable, pickle_out)
    # Close the file
    pickle_out.close()


# In[ ]:


def pickle_in_variable(filename, folder_path='.'):
    """
    Simple function to pickle in a single python variable
    given the filename (filename.pickle), as a string
    and a foldername ('.' for current directory) as a string
    """
    # Open the file
    pickle_in = open(os.path.join(folder_path, filename), 'rb')
    # Read in the variable
    variable = pickle.load(pickle_in)
    # Close the file
    pickle_in.close()

    # Return the variable
    return variable


# ## Function to injure weights

# In[ ]:


def define_order_of_injury(weights):
    '''
    Define the order in which progressive injury will be applied.
    
    Input: Model weights as list of numpy arrays
        - Mainly for defining the size of order-of-injury list

    Output: Indices by which to order the injury mask, as a list of numpy arrays
        - Bias is ignored and passed as unrandomized indices
        - List element for weights is a randomized 1-d numpy array
         with values from 0 to (# elements - 1)
        - This serves as part of the input when creating the injury mask
            - When creating the injury mask of a given layer, instead of
              randomizing the injury mask array, re-arrange them according
              to this randomized indices. Thus, the ordered injury mask
              will always be randomized the same way according to a
              once-defined randomization.
            - You can then increase the amount of injury in a given layer
              and distributed that randomly to the weights in the layer
              in a consistent manner.
    '''
    
    injury_order = [None]*len(weights)
    
    
    for i in range(len(injury_order)):
        # Initialize the indices to the size of the list element
        injury_order[i] = np.arange(weights[i].size)
        
        if i%2 == 0:  # For even indices, which are the weights
            # Randomly shuffle the indices of the weight elements
            np.random.shuffle(injury_order[i])
    
    return injury_order


# # Make injury map

# In[ ]:


injury_map = define_order_of_injury(weights)


# # Save injury map

# In[ ]:


pickle_out_variable(injury_map, injury_map_path)

