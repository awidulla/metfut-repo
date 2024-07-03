#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiple classes/functions copied from WeatherBench publication (Rasp et al.)
https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py

@author: awidulla
"""
# import a bunch of packages - don't worry about TF warnings
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
#import seaborn as sns
import pickle
#from src.score import *
from collections import OrderedDict

@keras.utils.register_keras_serializable()
class PeriodicPadding2D(tf.keras.layers.Layer):
    '''
    source: WeatherBench
    '''
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def call(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = tf.concat(
            [inputs[:, :, -self.pad_width:, :], inputs, inputs[:, :, :self.pad_width, :]], axis=2)
        # Zero padding in the lat direction
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]])
        return inputs_padded

    def get_config(self):
        config = super().get_config()
        config.update({'pad_width': self.pad_width})
        return config

@keras.utils.register_keras_serializable()
class PeriodicConv2D(tf.keras.layers.Layer):
    '''
    source: WeatherBench
    '''
    def __init__(self, filters,
                 kernel_size,
                 conv_kwargs={},
                 **kwargs, ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        if type(kernel_size) is not int:
            assert kernel_size[0] == kernel_size[1], 'PeriodicConv2D only works for square kernels'
            kernel_size = kernel_size[0]
        # kernel_size needs to be uneven!!!! - otherwise size decreases
        pad_width = (kernel_size - 1) // 2
        self.padding = PeriodicPadding2D(pad_width)
        self.conv = Conv2D(
            filters, kernel_size, padding='valid', **conv_kwargs
        )

    def call(self, inputs):
        return self.conv(self.padding(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'conv_kwargs': self.conv_kwargs})
        return config

# outside of Class
def build_cnn(filters, kernels, input_shape, dr=0):
    """
    Fully convolutional network
    source: WeatherBench
    """
    x = input = Input(shape=input_shape)
    for f, k in zip(filters[:-1], kernels[:-1]):
        x = PeriodicConv2D(f, k)(x)
        x = LeakyReLU()(x)
        if dr > 0: x = Dropout(dr)(x)
    output = PeriodicConv2D(filters[-1], kernels[-1])(x)
    return keras.models.Model(input, output)

'''
Everything below written by myself
'''

def initialize_model(data, N_layers, Kernel_size, N_filters):
    
    # create input variables for build_cnn
    filters     = [N_filters]*(N_layers+1)
    filters[-1] = data.output_filters
    kernels     = [Kernel_size]*(N_layers+1)
    
    print(filters, kernels, data.input_shape)
    
    # build and compile
    model = build_cnn(filters, kernels, data.input_shape)
    model.compile(keras.optimizers.Adam(1e-4), loss=data.custom_loss)
    
    return model

def initialize_docu_model(data):
    
    # create dictionary
    documentation = {}
    
    documentation['variables'] = data.output_variables
    documentation['training']  = []
    documentation['time']      = []
    
    documentation['RMSE']      = {}
    documentation['RMSE']['train'] = []
    documentation['RMSE']['valid'] = []
    
    for i in range(data.output_filters):
        documentation['RMSE']['train'].append([])
        documentation['RMSE']['valid'].append([])
    
    return documentation

def training_cycle(model, data, N_epochs, documentation):
    
    # time keeping for training cycle
    t_start = time.time()
    
    # train model for given number of epochs
    model.fit(data.X_train, data.Y_train, batch_size=128, epochs=N_epochs)
    # document training
    documentation['training'].append(N_epochs)
    
    # compute prediction for train and validation data_set
    pred_train = model.predict(data.X_train)
    pred_valid = model.predict(data.X_valid)
    
    # apply mask and compute residuals
    e_train = pred_train*data.surface_mask-data.Y_train*data.surface_mask
    e_valid = pred_valid*data.surface_mask-data.Y_valid*data.surface_mask
    
    # compute RMSE seperately for each output field
    for v in range(data.output_filters):
        rmse_train = np.sqrt(np.mean(np.square(e_train[:,:,:,v])))
        rmse_valid = np.sqrt(np.mean(np.square(e_valid[:,:,:,v])))
        # document RMSE
        documentation['RMSE']['train'][v].append(rmse_train)
        documentation['RMSE']['valid'][v].append(rmse_valid)
    
    # save time keeping
    documentation['time'].append(time.time()-t_start)

def hyper_training(data):
    '''
    Initialize and train a model for all variations of hyper parameters
    All models are saved, training process is saved in dictionary
    '''
    
    # create dictionary for documenting all relevant parameters
    docu = {}
    docu['model_names'] = []
    
    docu['hyper'] = {}
    docu['hyper']['layers']  = []
    docu['hyper']['kernels'] = []
    docu['hyper']['filters'] = []
    
    # iterate all hyperparameters
    for l in data.layers:
        for k in data.kernels:
            for f in data.filters:
                
                # skip the execution of first three models - memory problem
                if l==3 and k==3:
                    continue
                
                if l==3 and k==5 and f==32:
                    continue
                
                # creating distinct name coding
                model_name = data.case_type+'_s'+str(data.lead_steps)+'_l'+\
                             str(l)+'_k'+str(k)+'_f'+str(f)
                
                print(model_name)
                docu['model_names'].append(model_name)
                docu['hyper']['layers'].append(l)
                docu['hyper']['kernels'].append(k)
                docu['hyper']['filters'].append(f)
                
                # initialize model
                model = initialize_model(data, l, k, f)
                model.summary()
                
                # initialize documentation
                docu[str(model_name)] = initialize_docu_model(data)
                
                # run training cycles
                for c in range(data.cycles):
                    training_cycle(model, data, data.epochs, docu[str(model_name)])
                    
                # save model
                model.save(data.rootdir+'Model/'+model_name+'.keras')
    
                # save training for each model (in case server breaks)
                with open(data.rootdir+'Model/'+model_name+'.pickle', 'wb') as file:
                    pickle.dump(docu[str(model_name)], file)
                
                # clear RAM
                K.clear_session()
    
    # save training for all models
    with open(data.rootdir+'Model/'+data.case_type+'_s'+str(data.lead_steps)
              +'.pickle', 'wb') as file:
        pickle.dump(docu[str(model_name)], file)
    

def main():
    root_era5 = '/home/anton/Documents/MSPEA/SoSe24/MachineLearning/data_retrieval/2024METFUT_project_data/ERA5/'
    data_ocean = data_prep(root_era5, 5)
    data_ocean.ocean()
    
    model = initialize_model(data_ocean, 1, 32, 5)
    documentation = initialize_documentation(data_ocean)
    return data_ocean, model, documentation

def import_model(path, data):
    # load
    model = keras.models.load_model(path)
    
    # recompile
    model.compile(keras.optimizers.Adam(1e-4), loss=data.custom_loss)

    return model 

@keras.utils.register_keras_serializable()
class CustomLoss(keras.losses.Loss):
    def __init__(self, surface_mask, reduction=tf.keras.losses.Reduction.AUTO, name='custom_loss'):
        super().__init__(reduction=reduction, name=name)
        self.surface_mask = surface_mask

    def call(self, y_true, y_pred):
        # ensure surface_mask has the same shape as y_true and y_pred
        #self.surface_mask = tf.broadcast_to(self.surface_mask, tf.shape(y_true))
        
        # apply surface mask to data
        y_true = y_true*self.surface_mask
        y_pred = y_pred*self.surface_mask
        
        # compute loss
        loss = K.sqrt(K.mean(K.square(y_true-y_pred)))
        
        return loss        

    def get_config(self):
        config = super().get_config()
        config.update({'surface_mask': self.surface_mask})
        return config



class data_prep:
    def __init__(self, rootdir, lead_steps):
        self.rootdir      = rootdir   # path of project data folder 
        self.lead_steps   = lead_steps
        self.case_type    = None
        
        # default time periods used by Weatherbench 
        self.train_years  = slice('1979', '2015')
        self.valid_years  = slice('2016', '2016')
        self.test_years   = slice('2017', '2018')
        
        # different datasets
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test  = None
        self.Y_test  = None
        
        # parameters that might be of interest
        self.input_shape      = None
        self.output_filters   = None
        
        self.input_variables  = None
        self.output_variables = None
        
        # normalization parameters that were applied to datasets
        self.mean = None
        self.std  = None
        
        # climatology
        self.climatology = None
        
        # mask relevant for the loss function
        self.surface_mask = None
        self.custom_loss  = None
        
        # set hyperparameters that will be iterated
        self.layers  = [3,4,5]
        # kernels need to be uneven
        self.kernels = [3,5,7] 
        self.filters = [32,64,128]
        
        # set number of epochs per training cycle and number of cycles
        self.epochs = 5
        self.cycles = 30
        
        
    def change_time(self, begin_train, begin_valid, begin_test, end_test):
        '''
        Function to change default time ranges
        '''
        self.train_years  = slice(str(begin_train), str(begin_valid))
        self.valid_years  = slice(str(begin_valid), str(begin_test))
        self.test_years   = slice(str(begin_test),  str(end_test))
    
    def import_ocean(self):
        '''
        Import ocean fields and replace nan values with climatology
        '''
        # load relevant datasets
        sst    = xr.open_mfdataset(self.rootdir+'ERA5/sst_5.625deg/*.nc',
                                   combine='by_coords').load()
        siconc = xr.open_mfdataset(self.rootdir+'ERA5/siconc_5.625deg/*.nc',
                                   combine='by_coords').load()
        
        # compute climatology
        sst_clim    = sst.groupby('time.dayofyear').mean()
        siconc_clim = siconc.groupby('time.dayofyear').mean()
        
        # fill nan values using climatologies - otherwise no learning possible
        sst = sst.fillna(sst_clim.sel(dayofyear=sst.time.dt.dayofyear))
        siconc = siconc.fillna(siconc_clim.sel(dayofyear=siconc.time.dt.dayofyear))
        
        return sst, siconc
    
    '''
    write functions
    import_atmosphere
    import_surface
    
    import_masks # call them sea, land, all
    '''
    # Naming
    # Output_Input 
    # O - Ocean
    # L - Land
    # A - Atmosphere
    
    def O_O(self):
        '''
        Function that imports and sets up all relevant data for an ocean model
        '''
        # document that ocean was called
        self.case_type = 'O_O'
        
        # import data
        sst, siconc = self.import_ocean()
        
        # create datasets
        self.get_input_dataset([sst, siconc], ['sst', 'siconc'])
        self.get_output_dataset([sst, siconc], ['sst', 'siconc'])
        
        self.input_variables  = ['sst', 'siconc']
        self.output_variables = ['sst', 'siconc']
        
        # load land sea mask and create a sea-land mask
        lsmask = xr.open_dataset(self.rootdir+'ERA5/lsm_5.625deg.nc').load()
        slm = xr.where(lsmask>0.5,0,1)['lsm'].values[0,:,:]
        
        self.surface_mask = K.constant(np.stack([slm, slm],axis=2))
        self.custom_loss  = CustomLoss(self.surface_mask)
    
    # Maybe split this into function that creates input and output data
    def get_input_dataset(self, datasets, variables):
        # create lists
        X_train = []
        X_valid = []
        X_test  = []

        # iterate over index
        for i, d in enumerate(datasets):
          
          # Split train, valid and test dataset
          train_d = d.sel(time=self.train_years)
          valid_d = d.sel(time=self.valid_years)
          test_d  = d.sel(time=self.test_years)
          
          # Normalize the data using the mean and standard deviation of the training data
          mean = d.mean()[variables[i]].values
          std  = d.std()[variables[i]].values
          
          train_d = (train_d-mean)/std
          valid_d = (valid_d-mean)/std
          test_d  = (test_d-mean)/std
          
          # Create inputs and outputs that are shifted by lead_steps
          X_train.append(getattr(train_d, variables[i]).isel(time=slice(None, -self.lead_steps)).values[..., None])
          X_valid.append(getattr(valid_d, variables[i]).isel(time=slice(None, -self.lead_steps)).values[..., None])
          X_test.append(getattr(test_d, variables[i]).isel(time=slice(None, -self.lead_steps)).values[..., None])
          
        # concatenate training data for multi layer input
        self.X_train = np.concatenate(X_train, axis = 3)
        self.X_valid = np.concatenate(X_valid, axis = 3)
        self.X_test  = np.concatenate(X_test, axis = 3)
        
        # extract input shape
        self.input_shape = self.X_train.shape[1:]
    
    def get_output_dataset(self, datasets, variables):
        # create lists
        Y_train = []
        Y_valid = []
        Y_test  = []
        
        # save mean and std to later denormalize results
        self.mean = np.zeros(len(datasets))
        self.std  = np.zeros(len(datasets))
          
        # iterate over index
        for i, d in enumerate(datasets):
          
          # Split train, valid and test dataset
          train_d = d.sel(time=self.train_years)
          valid_d = d.sel(time=self.valid_years)
          test_d  = d.sel(time=self.test_years)
          
          # Normalize the data using the mean and standard deviation of the training data
          self.mean[i] = d.mean()[variables[i]].values
          self.std[i]  = d.std()[variables[i]].values
          
          train_d = (train_d - self.mean[i]) / self.std[i]
          valid_d = (valid_d - self.mean[i]) / self.std[i]
          test_d  = (test_d  - self.mean[i]) / self.std[i]
          
          # Create inputs and outputs that are shifted by lead_steps
          Y_train.append(getattr(train_d, variables[i]).isel(time=slice(self.lead_steps, None)).values[..., None])
          Y_valid.append(getattr(valid_d, variables[i]).isel(time=slice(self.lead_steps, None)).values[..., None])
          Y_test.append(getattr(test_d, variables[i]).isel(time=slice(self.lead_steps, None)).values[..., None])
          
        # concatenate training data for multi layer input
        self.Y_train = np.concatenate(Y_train, axis = 3)
        self.Y_valid = np.concatenate(Y_valid, axis = 3)
        self.Y_test  = np.concatenate(Y_test, axis = 3)
        
        # information for shape of output
        self.output_filters = self.Y_train.shape[-1]
    
    '''
    def custom_loss(self, y_true, y_pred):
        # apply surface mask to data
        y_true = y_true*self.surface_mask
        y_pred = y_pred*self.surface_mask

        return K.sqrt(K.mean(K.square(y_true-y_pred)))
    '''
    
if __name__ == '__main__':
    print('hello')
    #root_era5  = '/home/anton/Documents/MSPEA/SoSe24/MachineLearning/data_retrieval/2024METFUT_project_data/ERA5/'
    #data_ocean = data_prep(root_era5, 5)
    #data_ocean= data_ocean.ocean()