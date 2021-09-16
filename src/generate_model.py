import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from functools import partial



def __generate_1DCNN_model(feature_num, best_trial):
    """
    generate 1DCNN model
    """
    #############################################
    num_layer = best_trial['params_num_layer']
    num_linear = best_trial['params_num_linear']
    conv_units = [int(best_trial['params_conv_units'+str(i)]) for i in range(best_trial['params_num_layer'])]
    k_size = 3
    pad_size = int(best_trial['params_pad_size'])
    activation_name = best_trial['params_activation'] 
    linear_units = [int(best_trial['params_linear_units'+str(i)]) for i in range(best_trial['params_num_linear'])]
    pooling_name = best_trial['params_pooling']
    #############################################
        
    if activation_name == 'ReLU':
        activation = F.relu
    elif activation_name == 'Leaky_ReLU':
        activation = F.leaky_relu    
        
    if pooling_name == 'Max':
        pooling_layer = F.max_pooling_1d
    elif pooling_name == 'Average':
        pooling_layer = F.average_pooling_1d
        
    layers = []
    layers.append(L.Convolution1D(in_channels=feature_num, out_channels=conv_units[0], ksize=k_size, pad=pad_size))
    layers.append(L.BatchNormalization(conv_units[0]))
    if activation_name == 'Swish':
        layers.append(L.Swish(None))
    else:
        layers.append(activation)
    
    for i in range(1, num_layer):
        layers.append(L.Convolution1D(in_channels=conv_units[i-1], out_channels=conv_units[i], ksize=k_size, pad=pad_size))
        layers.append(L.BatchNormalization(conv_units[i]))
        if activation_name == 'Swish':
            layers.append(L.Swish(None))
        else:
            layers.append(activation)
    
    layers.append(partial(pooling_layer, ksize=2))
    
    for i in range(num_linear):
        layers.append(L.Linear(None, linear_units[i]))
        layers.append(L.BatchNormalization(linear_units[i]))
        if activation_name == 'Swish':
            layers.append(L.Swish(None))
        else:
            layers.append(activation)
    
    layers.append(L.Linear(None, 1))
    
    return chainer.Sequential(*layers)




# CyclicConv
def pre(data, residue_count, number):
    list_ = []
    for i in range(len(data)):
        x = data[i]
        residue_ = residue_count[i]
        number_ = number[i]
        pad_left = -1
        pad_right = max_len
        # padding
        for j in range(max_len-1):
            number_now = number_[j]
            number_next = number_[j+1]

            if (number_now != -1) and (j == 0):
                pad_right = residue_
                break
            else:
                if (number_now == -1) and (number_next != -1):
                    pad_left = j
                    if j+residue_ == max_len-1:
                        pad_right = max_len
                        break
                elif (number_now != -1) and (number_next == -1):
                    pad_right = j+1

        left_ = x[:,:pad_left+1]
        right_ = x[:,pad_right:]

        mid_ = x[:,pad_left+1:pad_right]
        mid_ = chainer.functions.concat([mid_[:,-1:], mid_, mid_[:,:1]], axis=1)

        tmp = chainer.functions.concat([left_, mid_, right_], axis=1)

        list_.append(tmp.data) 
    list_ = np.array(list_)
    return Variable(list_) 
    
    

class CyclicConv(chainer.Chain):
    def __init__(self, num_layer, num_linear, conv_units, linear_units, k_size, pad_size, activation_name, pooling_name):
        super().__init__()
        with self.init_scope():
            self.num_layer = num_layer
            self.num_linear = num_linear
            self.activation_name = activation_name
            self.pooling_name = pooling_name
            
            # layer_0
            self.conv0 = L.Convolution1D(in_channels=feature_num, out_channels=conv_units[0], ksize=k_size, pad=pad_size)    
            self.bnconv0 = L.BatchNormalization(conv_units[0])    
            if activation_name == 'Swish':
                self.sconv0 = L.Swish(None)
                
            # layer_1
            if num_layer >= 2:
                self.conv1 = L.Convolution1D(in_channels=conv_units[0], out_channels=conv_units[1], ksize=k_size, pad=pad_size)
                self.bnconv1 = L.BatchNormalization(conv_units[1]) 
                if activation_name == 'Swish':
                    self.sconv1 = L.Swish(None)
            # layer_2
            if num_layer >= 3:
                self.conv2 = L.Convolution1D(in_channels=conv_units[1], out_channels=conv_units[2], ksize=k_size, pad=pad_size)
                self.bnconv2 = L.BatchNormalization(conv_units[2]) 
                if activation_name == 'Swish':
                    self.sconv2 = L.Swish(None)
            # layer_3
            if num_layer >= 4:
                self.conv3 = L.Convolution1D(in_channels=conv_units[2], out_channels=conv_units[3], ksize=k_size, pad=pad_size)
                self.bnconv3 = L.BatchNormalization(conv_units[3]) 
                if activation_name == 'Swish':
                    self.sconv3 = L.Swish(None)
            # layer_4
            if num_layer >= 5:
                self.conv4 = L.Convolution1D(in_channels=conv_units[3], out_channels=conv_units[4], ksize=k_size, pad=pad_size)
                self.bnconv4 = L.BatchNormalization(conv_units[4]) 
                if activation_name == 'Swish':
                    self.sconv4 = L.Swish(None)


            # dense
            self.l0 = L.Linear(None, linear_units[0]) 
            self.bnl0 = L.BatchNormalization(linear_units[0])
            if activation_name == 'Swish':
                self.sl0 = L.Swish(None)
            
            if num_linear >= 2:
                self.l1 = L.Linear(None, linear_units[1])
                self.bnl1 = L.BatchNormalization(linear_units[1])
                if activation_name == 'Swish':
                    self.sl1 = L.Swish(None)
            
            if num_linear >= 3:
                self.l2 = L.Linear(None, linear_units[2])
                self.bnl2 = L.BatchNormalization(linear_units[2])
                if activation_name == 'Swish':
                    self.sl2 = L.Swish(None)
            
            self.out = L.Linear(None, 1) 
            

    def forward(self, x, residue_count, number):

        h = pre(x, residue_count, number)
        h = self.bnconv0(self.conv0(h))
    
        if self.activation_name == 'Swish':
            h = self.sconv0(h)
        elif self.activation_name == 'ReLU':
            h = F.relu(h)
        elif self.activation_name == 'Leaky_ReLU':
            h = F.leaky_relu(h)

        if self.num_layer >= 2:
            h = pre(h, residue_count, number)
            h = self.bnconv1(self.conv1(h))
            if self.activation_name == 'Swish':
                h = self.sconv1(h)
            elif self.activation_name == 'ReLU':
                h = F.relu(h)
            elif self.activation_name == 'Leaky_ReLU':
                h = F.leaky_relu(h)
 
        if self.num_layer >= 3:
            h = pre(h, residue_count, number)
            h = self.bnconv2(self.conv2(h))
            if self.activation_name == 'Swish':
                h = self.sconv2(h)
            elif self.activation_name == 'ReLU':
                h = F.relu(h)
            elif self.activation_name == 'Leaky_ReLU':
                h = F.leaky_relu(h)
            
        if self.num_layer >= 4:    
            h = pre(h, residue_count, number)
            h = self.bnconv3(self.conv3(h))
            if self.activation_name == 'Swish':
                h = self.sconv3(h)
            elif self.activation_name == 'ReLU':
                h = F.relu(h)
            elif self.activation_name == 'Leaky_ReLU':
                h = F.leaky_relu(h)
            
        if self.num_layer >= 5:     
            h = pre(h, residue_count, number)
            h = self.bnconv4(self.conv4(h))
            if self.activation_name == 'Swish':
                h = self.sconv4(h)
            elif self.activation_name == 'ReLU':
                h = F.relu(h)
            elif self.activation_name == 'Leaky_ReLU':
                h = F.leaky_relu(h)
                
            
        if self.pooling_name == 'Max':
            h = F.max_pooling_1d(h, ksize=2)
        elif self.pooling_name == 'Average':
            h = F.average_pooling_1d(h, ksize=2)
        
        
        h = self.bnl0(self.l0(h))
        if self.activation_name == 'Swish':
            h = self.sl0(h)
        elif self.activation_name == 'ReLU':
            h = F.relu(h)
        elif self.activation_name == 'Leaky_ReLU':
            h = F.leaky_relu(h)
        
        if self.num_linear >= 2:
            h = self.bnl1(self.l1(h))
            if self.activation_name == 'Swish':
                h = self.sl1(h)
            elif self.activation_name == 'ReLU':
                h = F.relu(h)
            elif self.activation_name == 'Leaky_ReLU':
                h = F.leaky_relu(h)
                
        if self.num_linear >= 3:
            h = self.bnl2(self.l2(h))
            if self.activation_name == 'Swish':
                h = self.sl2(h)
            elif self.activation_name == 'ReLU':
                h = F.relu(h)
            elif self.activation_name == 'Leaky_ReLU':
                h = F.leaky_relu(h)      
        
            
        h = self.out(h)

        return h


def __generate_CyclicConv_model(feature_num, best_trial):
    """
    generate CyclicConv model
    """
    #############################################
    num_layer = trial['params_num_layer']
    num_linear = trial['params_num_linear']
    conv_units = [int(trial['params_conv_units'+str(i)]) for i in range(trial['params_num_layer'])]
    linear_units = [int(trial['params_linear_units'+str(i)]) for i in range(trial['params_num_linear'])]
    k_size = 3
    pad_size = 0  
    activation_name = trial['params_activation'] 
    pooling_name = trial['params_pooling']
    #############################################
    
    if activation_name == 'ReLU':
        activation = F.relu
    elif activation_name == 'Leaky_ReLU':
        activation = F.leaky_relu

    if pooling_name == 'Max':
        pooling_layer = F.max_pooling_1d
    elif pooling_name == 'Average':
        pooling_layer = F.average_pooling_1d

    return CyclicConv(num_layer, num_linear, conv_units, linear_units, k_size, pad_size, activation_name, pooling_name)




def __generate_prediction_model(feature_num=3, 
                                use_augmentation=True,
                                use_CyclicConv=False,
                                weight_path="model_weight/model.npz"
                               ):
    """
    generate prediction model
    Parameters such as params_conv_units0 are set when using the rdkit descriptor, 
    so they differ from the results of Table S4 in the supplementary information using the moe descriptor.
    """
    if use_augmentation:
        if use_CyclicConv:
            best_trial={}
        else:
            best_trial={
                        'params_activation': 'ReLU',
                        'params_conv_units0': 55,
                        'params_conv_units1': 37,
                        'params_conv_units2': 174,
                        'params_conv_units3': 204,
                        'params_linear_units0': 166,
                        'params_linear_units1': 33,
                        'params_linear_units2': 43,
                        'params_num_layer': 4,
                        'params_num_linear': 3,
                        'params_pad_size': 0,
                        'params_pooling': 'Average'
                        }
    else:
        if use_CyclicConv:
            best_trial={}
        else:
            best_trial={}

    if use_CyclicConv:
        model = __generate_CyclicConv_model(feature_num, best_trial)
    else:
        model = __generate_1DCNN_model(feature_num, best_trial)

    chainer.serializers.load_npz(weight_path, model) 
    
    return model


