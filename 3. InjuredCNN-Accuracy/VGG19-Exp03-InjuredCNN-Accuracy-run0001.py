#!/usr/bin/env python
# coding: utf-8

# # Simulate progressive neurodegeneration in trained VGG19 CNN and evaluate accuracy on CIFAR100
# 
# Use the trained "healthy" model and defined injury map to simulate progressive neurodegeneration by synaptic ablation on a CNN. Evaluate performance at the class and superclass level on CIFAR100 over the course of neurodegeneration by calculating:
# - Class-wise classification accuracy
# - Superclass-wise classification accuracy
# - "Errors within the correct superclass": What percentage of misclassifications at the class level were still correct at the superclass level (e.g. a rose misclassified as a tulip, another member of the superclass flower)

# # Input Parameters

# ## Injury Levels
# How much the network will be injured at each step in the simulation. 
# 
# The length of the list *injury_levels* indicates how many injury steps will be performed, and the corresponding value at that list location is the percentage of weights to be injured in each layer (expressed as a fraction)

# In[ ]:


# Run to 50% injury with 0.1% increments in number of weights injured
injury_levels = [x/1000 for x in range(0,501,1)]  # start, end, step 


# ## Run ID

# In[ ]:


# Modifiable value of the run ID
run = '0001'


# ## Model filepath

# In[ ]:


trained_dir = '../Trained_Models'
trained_prefix = '/VGG19-Exp01-TrainedModel-run'
trained_suffix = '.h5'

# Filename
model_filename = trained_dir + trained_prefix + str(run) + trained_suffix
print("Model:\n\t",model_filename)


# ## Injury map filepath

# In[ ]:


map_dir = '../Injury_Maps'
map_prefix = '/VGG19-Exp02-InjuryMap-run'
map_suffix = '.pickle'

# Filename
map_filename = map_dir + map_prefix + str(run) + map_suffix
print("Injury map:\n\t",map_filename)


# ## Output filename and path

# In[ ]:


save_dir = './RESULTS'
save_prefix = '/VGG19-Exp03-InjuredCNN-Accuracy-run'
save_suffix = '.csv'

save_filename = save_dir + save_prefix + str(run) + save_suffix

print("Save to:\n\t", save_filename)


# ## Path to class to superclass conversion
# A pickled list that maps the class label of CIFAR100 to its corresponding superclass

# In[ ]:


path_to_class_superclass_conversion = "../Data/finelabel_to_coarselabel_conversionlist.pickle"


# # Load modules

# In[ ]:


import tensorflow.keras as K
import numpy as np
import os
import pickle
import sklearn.metrics
import pandas as pd


# # Define functions

# ## Function to pre-process data for VGG19 model

# In[ ]:


def preprocess_data(X, Y):
    """Pre-processes the data for the trained model"""
    X_p = K.applications.vgg19.preprocess_input(X.astype('float32')) # pre-processed input
    Y_p = K.utils.to_categorical(Y, classes) # one-hot matrix
    return X_p, Y_p


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


def apply_progressive_injury(initial_weights, injury_order, injury_amount):
    """
    Given (3):
        initial weights: A set of (healthy) weights
        injury_order: Order of weights to injure for each layer.
        injury_amount: Proportion of weights (0 to 1) to injure in each layer
    
    Apply an injury (setting weights to 0) to injury_amount*100 % of WEIGHTS
    in a given layer in a fixed-random fashion.
    
    Fixed because the order in which the weights are injured is defined
    elsewhere and passed in.
    
    The biases are not modified
    
    Returns (2):
        final_weights: updated value of weights with a given injury applied
            to be loaded back into the model
        injury_mask: information on whether a given weight is injured [0] or 
            unmodified [1]. Same structure as the initial_weights and 
            final_weights (lists with numpy arrays)
    """
    
    ## Initialize the injury mask to multiply the initial weights with
    ##. A list of the same size as the weights, containing 0's and 1's
    injure_mask = [None]*len(initial_weights)
    
    for i in range(len(injure_mask)):
        if initial_weights[i].ndim == 1:  # Non-weight indices
            # If this is a bias layer or batch norm layer
            # Just make a list of 1's of the given size
            x = [1]*initial_weights[i].size
            
            # NOTE:
            ## If you want to injure biases, then modify the script here
            ## You will also need to modify the injury_order defined
            ## in "2. CreateInjuryMap"
        
        else:  # For weights
            # If not a bias layer, then assume weight layer
            # which is true for VGG19.
            # NOTE: This would need to be modified for other networks

            # Injure x% of weights in this layer

            # Total number of weights in the layer
            num_weights = initial_weights[i].size
            
            # Number of weights to spare
            num_to_spare = round(num_weights*(1-injury_amount))
            # If rounding results in no weights spared, set to minimum 1
            if num_to_spare < 1:
                num_to_spare = 1
                
            # Number of weights to injure
            num_to_injure = num_weights - num_to_spare
            
            # Create numpy array with the correct number of injured
            # and spared weights (in that order)
            ## The order of this 1-d array will need to be modified
            x = np.array([0]*num_to_injure + [1]*num_to_spare)  # np array
            
            # Then, re-order/shuffle this list (stored as np array)
            # according to a previously-defined randomized order
            # defined in injury_order[i]
            x = x[injury_order[i]]
        
        # Reshape x to match this given list element (either weight or bias)
        x = np.reshape(x, initial_weights[i].shape)  # np array
        # Assign to the injure_list mask element
        injure_mask[i] = x  # list, with element as np array
    
    # Create the list of injured weights
    # by multiplying the initial weights with the mask
    # As the bias elements of the mask are 1, biases are not modified
    final_weights = [None]*len(initial_weights)
    for w in range(0, len(initial_weights)):
        final_weights[w] = initial_weights[w]*injure_mask[w]

        
    return final_weights, injure_mask


# # Load data
# - Previously trained model
# - Previously created injury map
# - CIFAR100 images and label information (class and superclass)

# ## Load Model

# In[ ]:


model = K.models.load_model(model_filename)


# In[ ]:


model.summary()


# ### Get Healthy Weights

# In[ ]:


weights_healthy = model.get_weights()


# ## Load Injury Map

# In[ ]:


injury_map = pickle_in_variable(map_filename)


# ## Load CIFAR100 information
# 

# In[ ]:


# Load the raw cifar100 dataset
(raw_X_train, raw_Y_train), (raw_X_test, raw_Y_test) = K.datasets.cifar100.load_data()
classes = 100


# In[ ]:


# Pre-process the training and validation data
X_test, Y_test = preprocess_data(raw_X_test, raw_Y_test)


# In[ ]:


# Load the list that maps each class label to its corresponding superclass label
convert_class_to_super = pickle_in_variable(path_to_class_superclass_conversion)


# In[ ]:


# Ground truths as list
gt_class = [raw_Y_test[i][0] for i in range(len(raw_Y_test))]


# In[ ]:


# Superclass ground truths as list
gt_super = [convert_class_to_super[i] for i in gt_class]


# # Run the experiment
# Loop through the injury levels. At each injury level:
# - Injure the model by ablating the given % of model weights
#     - Create a set of injured weights
#     - Load into the model
# - Evaluate the model on the test set, return:
#     - Class accuracy
#     - Superclass accuracy
#     - Class errors within the correct superclass

# In[ ]:


# Number of injury steps
num_inj = len(injury_levels)


# In[ ]:


# Initialize lists to store results
acc_overall = [-1.]*num_inj  # Class accuracy
acc_super = [-1.]*num_inj  # Superclass accuracy
err_correct_super = [-1.]*num_inj  # Errors within the correct superclass


# In[ ]:


# Loop Through injury levels
for i in range(num_inj):
    # Set injury level for this loop
    inj = injury_levels[i]
    
    print("Inj:", inj)
    
    # Get injured weights
    weights_injured, mask_injured = apply_progressive_injury(
        initial_weights = weights_healthy,
        injury_order = injury_map,
        injury_amount = inj
    )
    
    # Set model with injured weights
    model.set_weights(weights_injured)
    
    # Get model class predictions
    predicted_class = model.predict_classes(X_test, batch_size=500, verbose=1)
    # Convert predictions to a list
    y_class = list(predicted_class)
    # Get corresponding superclass of predictions
    y_super = [convert_class_to_super[k] for k in y_class]
    
    # Evaluate
    
    # Overall accuracy
    acc_overall[i] = sklearn.metrics.accuracy_score(gt_class, y_class)
    
    # Superclass accuracy
    acc_super[i] = sklearn.metrics.accuracy_score(gt_super, y_super)
    
    # Errors in superclass (fraction)
    ## Get list of superclass predictions of wrong fine-class predictions
    wrong_y_super = list()
    wrong_gt_super = list()
    for k in range(len(y_class)):
        if y_class[k] != gt_class[k]:  # Not equal
            wrong_y_super.append(y_super[k])
            wrong_gt_super.append(gt_super[k])
    ## Calculate superclass accuracy
    ## Of errors, look at what fraction are in the correct superclass
    err_correct_super[i] = sklearn.metrics.accuracy_score(wrong_gt_super, wrong_y_super)
    
    print(
          "\tAcc:", acc_overall[i],
          "\tSuper Acc:", acc_super[i],
          "\tErr Corr Super:", err_correct_super[i]
         )


# # Save Results

# ## Make directory if needed

# In[ ]:


# If results directory doesn't exist, make it
if not os.path.exists(save_dir):
    print("RESULTS directory doesn't exist, creating one")
    os.makedirs(save_dir)
else:
    print('RESULTS directory exists, no action taken')


# ## Create results dataframe

# In[ ]:


# First make a dictionary of the results
results = dict(
    Injury_Amount = injury_levels,
    Overall_Accuracy = acc_overall,
    Superclass_Accuracy = acc_super,
    Errors_within_Correct_Superclass = err_correct_super
)


# In[ ]:


# Turn into a dataframe
results_df = pd.DataFrame(results)


# In[ ]:


# Save to CSV
results_df.to_csv(save_filename, index=False)

