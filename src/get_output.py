import generate_input
import generate_model
import draw_saliency_2Dmol
import chainer
import numpy as np
import pandas as pd

        
    
def __perform_prediction(model, aug_maps):
    """
    Original output of the model: Predicted value of %PPB, calculated Saliency Score.
    """
    
    # Variable
    inputs = chainer.Variable(aug_maps)
    
#     with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    with chainer.using_config('train', False):
        outputs = model(inputs)
        aug_pred = outputs.data
    
    target_var = inputs
    target_var.grad = None 
    
    model.cleargrads()
    
#     output_var = outputs
    output_var = chainer.functions.sum(outputs)
    output_var.backward(retain_grad=False)
    
    aug_saliency = target_var.grad
        
    # Saliency Score  
    ##################################################################
    aug_saliency = np.abs(aug_saliency)
#     min_ = aug_saliency.min(axis=None, keepdims=True)
#     max_ = aug_saliency.max(axis=None, keepdims=True)
#     aug_saliency = (aug_saliency - min_) / (max_ - min_)
#     aug_saliency = aug_saliency / max_
    ##################################################################
    
    return aug_pred, aug_saliency



def __calculate_average_predicted_value_of_replicas(aug_pred, aug_index):
    """
    Take the average of the replicas of the predictions
    """
    tmp_ = np.concatenate([aug_pred, np.expand_dims(aug_index, 1)], axis=1)
    df_tmp = pd.DataFrame(tmp_, columns=['pred', 'index'])
    grouped = df_tmp.groupby('index')
    aug_pred = np.array(grouped.mean()['pred']).reshape(-1, 1)
    return aug_pred
    
    
    
def __scaling_predicted_values(aug_pred, lower_limit=50, upper_limit=95):
    """
    Rounding predictions to within a range
    """
    tmp_ = []
    for pred in aug_pred:
        pred = np.max([pred[0], lower_limit])
        pred = np.min([pred, upper_limit])
        tmp_.append(round(pred, 2))
    return tmp_



def __summarize_saliency_score_of_replicas(aug_saliency, aug_nums, aug_index, aug_table, feature_num, output_format=1):
    """
    Aggregate the average saliency_score of replicas.
    If output_format=1, take the sum of features of feature_num dimension of Saliency Score and make it 1D.
    """
    aug_info = pd.DataFrame([aug_nums, aug_index], index=['nums', 'index']).T

    whole_saliency_ave = []

    for number_pep in list(set(aug_index)):

        # all replicas
        aug_index_now = [i for i in aug_info[aug_info['index'] == number_pep].index]
        aug_table_now = aug_table[aug_index_now]
        aug_saliency_now = aug_saliency[aug_index_now]

        table_org = np.delete(aug_table_now[0], np.argwhere(aug_table_now[0] == -1))
        # residue number
        nums_now = len(table_org)

        whole_saliency_now = {}
        whole_saliency_ave_now = {}

        for number_replica in range(len(aug_table_now)):
            table_replica = aug_table_now[number_replica]
            saliency_replica = aug_saliency_now[number_replica]

            residue_table_replica = np.delete(table_replica, np.argwhere(table_replica == -1))
            residue_saliency_replica = np.delete(saliency_replica, np.argwhere(table_replica == -1), axis=1)

            # Rotate until it matches table_org
            while (residue_table_replica == table_org).all() == False:
                residue_table_replica = np.hstack([residue_table_replica[1:], residue_table_replica[0]])
                residue_saliency_replica = np.hstack([residue_saliency_replica[:, 1:], 
                                                      residue_saliency_replica[:, 0].reshape(feature_num, 1)])


            for i in range(nums_now):
                tmp_index = residue_table_replica[i]

                if number_replica == 0:
                    whole_saliency_now[tmp_index] = residue_saliency_replica[:, i].reshape(feature_num, 1)
                else:
                    whole_saliency_now[tmp_index] = np.hstack([whole_saliency_now[tmp_index], 
                                                               residue_saliency_replica[:, i].reshape(feature_num, 1)])

        # average of all replica
        for i in range(nums_now):
            tmp_index = residue_table_replica[i]
            whole_saliency_ave_now[tmp_index] = np.average(whole_saliency_now[tmp_index], axis=1).reshape(feature_num, 1)
            # sum saliency Score
            if output_format == 1:
                whole_saliency_ave_now[tmp_index] = whole_saliency_ave_now[tmp_index].sum()
        
        tmp_ = []
        for i in range(nums_now):
            tmp_.append(whole_saliency_ave_now[i])
        
        whole_saliency_ave.append(tmp_)
    # list 
    return whole_saliency_ave



def __summarize_saliency_score(aug_saliency, aug_nums, aug_table, feature_num, output_format=1):

    whole_saliency = []

    for number_pep in range(len(aug_nums)):
        aug_table_now = aug_table[number_pep]
        aug_saliency_now = aug_saliency[number_pep]
        whole_saliency_now = {}

        for i in range(aug_nums[number_pep]):
            tmp_index = aug_table_now[i]
            if output_format == 1:
                whole_saliency_now[tmp_index] = aug_saliency_now[:, i].sum()
            else:
                whole_saliency_now[tmp_index] = aug_saliency_now[:, i].reshape(feature_num, 1)

        tmp_ = []
        for i in range(nums_now):
            tmp_.append(whole_saliency_now[i])        
        
        whole_saliency.appned(tmp_)
    # list 
    return whole_saliency






def predict(smiles_list, fig_path='saliency_figures/', use_augmentation=True, use_CyclicConv=False):
    """
    It takes the input data (list of smiles) and returns:
    Predicted value of %PPB (however, predictions below lower_limit% and above upper_limit% are rounded), 
    Saliency Score (contribution) of each substructure, 
    Color Score of each substructure Saliency Score (contribution) of each substructure, 
    color of each substructure and a heatmap of the Saliency Score overlaid on the 2D structure of the peptide (svg).
    """

    ################################################
    # Dimension of the features to be used
    feature_num = 3
    # Maximum residue length 
    max_len = 15    
    ################################################


    # generate input
    aug_maps, aug_nums, aug_index, aug_table, substructures_smiles, mapping_list, signed_mol_list = generate_input.__make_up_CNN_input(smiles_list, feature_num=feature_num, max_len=max_len, use_augmentation=use_augmentation)

    
    # make model
    model = generate_model.__generate_prediction_model(feature_num=feature_num, use_augmentation=use_augmentation, use_CyclicConv=use_CyclicConv, weight_path="model_weight/model.npz")
    
    
    aug_pred, aug_saliency = __perform_prediction(model, aug_maps)


    if use_augmentation:
        aug_pred = __calculate_average_predicted_value_of_replicas(aug_pred, aug_index)
        saliency = __summarize_saliency_score_of_replicas(aug_saliency, aug_nums, aug_index, aug_table, feature_num=feature_num, output_format=1)
    else: 
        saliency = __summarize_saliency_score(aug_saliency, aug_nums, aug_table, feature_num=feature_num, output_format=1)  


    # sacling
    pred = __scaling_predicted_values(aug_pred)   


    # draw 2D mol 
    draw_saliency_2Dmol.__visualization_of_saliency_score(fig_path, signed_mol_list, mapping_list, saliency)

    # color of substructures
    saliency_color_list = draw_saliency_2Dmol.__get_color_of_saliency_score(saliency)
    

    return pred, substructures_smiles, saliency, saliency_color_list






