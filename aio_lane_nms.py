import numpy as np
import torch


def Lane_nms(proposals, scores, overlap=50, top_k=4):
    keep_index = []
    # sorted_score = np.sort(scores)[-1] # from big to small 
    # indices = np.argsort(-scores) # from big to small 
    
    # r_filters = np.zeros(len(scores))

    sorted_score = torch.sort(scores)[-1] # from big to small 
    indices = torch.argsort(-scores) # from big to small 
    
    r_filters = torch.zeros(len(scores))

    for i, indice in enumerate(indices):
        if r_filters[i] == 1:  # continue if this proposal is filted by nms before
            continue
        keep_index.append(indice)
        if len(keep_index)>top_k:  # break if more than top_k
            break
        if i == (len(scores)-1):  # break if indice is the last one
            break
        sub_indices = indices[i+1:]
        for sub_i, sub_indice in enumerate(sub_indices):
            r_filter = Lane_IOU(proposals[indice,:],proposals[sub_indice,:],overlap)
            if r_filter: r_filters[i+1+sub_i]=1 
    num_to_keep = len(keep_index)
    keep_index = list(map(lambda x: x.item(), keep_index))
    return keep_index, num_to_keep


def Lane_IOU(parent_box, compared_box, threshold):
    '''
    calculate distance one pair of proposal lines
    return True if distance less than threshold 
    '''
    n_offsets = 72
    n_strips = n_offsets - 1

    start_a = (parent_box[2] * n_strips + 0.5).astype(int) # add 0.5 trick to make int() like round  
    start_b = (compared_box[2] * n_strips + 0.5).astype(int)
    start = max(start_a,start_b)
    end_a = start_a + parent_box[4] - 1 + 0.5 - (((parent_box[4] - 1)<0).astype(int))
    end_b = start_b + compared_box[4] - 1 + 0.5 - (((compared_box[4] - 1)<0).astype(int))
    end = min(min(end_a,end_b),71)
    if (end - start)<0:
        return False
    dist = 0
    for i in range(5+start,5 + int(end)):
        if i>(5+end):
                break
        if parent_box[i] < compared_box[i]:
            dist += compared_box[i] - parent_box[i]
        else:
            dist += parent_box[i] - compared_box[i]
    return dist < (threshold * (end - start + 1))
