#%%
import copy, random
import torch
from tqdm import tqdm
from typing import Union, List
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from get_representation import load_rep_label

def recenter(x, mean=None):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    return x - mean, mean

def project_onto_direction(H, direction, H_mean = None):
    mag = np.linalg.norm(direction)
    if H_mean is not None:
        H = H - H_mean
    assert not np.isinf(mag)
    X =  H.dot(direction) 
    X /= mag
    return X
    # return H.matmul(direction) / mag
    
def get_direction(hidden_states, train_labels, n_difference=2, n_components=2, is_recenter=False):
    """
        Args:
            hidden_states: (n_sample, hidden_dim)
            train_labels: (n_sample, 2)
            n_difference: int
            n_components: int
        Returns:
            directions: (n_components, hidden_dim)
            signs: (n_components,)
            importances: (n_components,)
            H_train_mean: (hidden_dim,)
    """
    # hidden_sates: (n_sample, hidden_dim)
    relative_hidden_states = copy.deepcopy(hidden_states)
    
    H_train_mean = None
    if is_recenter:
        relative_hidden_states, H_train_mean = recenter(relative_hidden_states)
    
    # obtain relative hidden sates
    for _ in range(n_difference):
        relative_hidden_states = relative_hidden_states[::2] - relative_hidden_states[1::2]

    H_train = relative_hidden_states

    # H_train_mean = None
    pca_model = PCA(n_components=n_components, whiten=False).fit(H_train)

    directions = pca_model.components_
    signs = get_signs(hidden_states=hidden_states, directions=directions, train_labels=train_labels, n_components=n_components)
    importances = pca_model.explained_variance_ratio_
    
    return directions, signs, importances, H_train_mean

def get_signs(hidden_states, directions,train_labels, n_components):
    signs = np.zeros(n_components)
    for component_index in range(n_components):
        transformed_hidden_states = project_onto_direction(hidden_states, directions[component_index])
        pca_output_max = 0
        pca_output_min = 0
        for i, label in enumerate(train_labels):
            if (label[0] == 1 and transformed_hidden_states[i * 2] > transformed_hidden_states[i * 2 + 1]):
                pca_output_max += 1
            elif (label[0] == 0 and transformed_hidden_states[i * 2] < transformed_hidden_states[i * 2 + 1]):
                pca_output_max += 1
            else:
                pca_output_min += 1
        signs[component_index] = np.sign(pca_output_max - pca_output_min)
        if signs[component_index] == 0:
            signs[component_index] = 1
    
    return signs

def select_train_data(eval_type, embeddings, train_labels, n_pair, high_bound=1, low_bound=0, is_random=True):
    if eval_type == "pairwise":
        high_bound, low_bound = 1, 0
    train_emb_true_index = [
        i for i, label in enumerate(train_labels) if label >= high_bound]
    train_emb_false_index = [
        i for i, label in enumerate(train_labels) if label <= low_bound]
    n_true = len(train_emb_true_index)
    n_false = len(train_emb_false_index)
    if n_pair > min(n_true, n_false):
        print(f"Warning: n_pair is larger than the number of true or false samples, n_pair is set to {min(n_true, n_false)}")
    if eval_type == "pairwise":
        assert n_true == n_false
    
    if is_random:
        if eval_type == "absolute":
            train_emb_true_index = random.sample(train_emb_true_index, n_pair)
            train_emb_false_index = random.sample(train_emb_false_index, n_pair)
        elif eval_type == "pairwise":
            # make sure the true/false samples of pairwise_eval are in pairs
            random_index = random.sample(list(range(n_true)), n_pair)
            train_emb_true_index = [train_emb_true_index[i] for i in random_index]
            train_emb_false_index = [train_emb_false_index[i] for i in random_index]
    else:
        train_emb_true_index = train_emb_true_index[:n_pair]
        train_emb_false_index = train_emb_false_index[:n_pair]

    train_emb_index = []
    train_emb = []
    for i in range(n_pair):
        train_emb_index.extend([train_emb_true_index[i],train_emb_false_index[i]])

    train_labels = [[1, 0]] * n_pair
    train_emb = embeddings[train_emb_index, :, :, :]
    return train_emb, train_labels

def get_eval_score_from_prediction(y_score, train_labels, eval_metric: Union[List[str], str] = "spearmanr"):
    if isinstance(eval_metric, str):
        eval_metric = [eval_metric]
    result = {}
    for metric in eval_metric:
        assert metric in ["spearmanr", "auc", "cross_entropy", "mse", "pair"]
        if metric == "spearmanr":
            correlation, p_value = spearmanr(y_score, train_labels)
            result[metric] = correlation
        elif metric == "auc":
            score = roc_auc_score(train_labels, y_score)
            result[metric] = score
        elif metric == "cross_entropy":
            score = -np.mean(train_labels * np.log(y_score) + (1 - train_labels) * np.log(1 - y_score))
            result[metric] = score
        elif metric == "mse":
            score = np.mean((train_labels - y_score) ** 2)
            result[metric] = score
        elif metric == "pair":
            length = len(train_labels) // 2
            score = 0
            for i in range(length):
                if (train_labels[i*2] > train_labels[i*2+1]) == (y_score[i*2] > y_score[i*2+1]):
                    score += 1
            result[metric] = score / length
    return result

def get_eval_score_of_single_direction(hidden_states, direction, sign, train_labels, H_mean = None, eval_metric: Union[List[str], str] = "spearmanr"):
    y_score = project_onto_direction(hidden_states, direction, H_mean=H_mean)
    y_score = np.dot(y_score, sign)
    return get_eval_score_from_prediction(y_score, train_labels, eval_metric)

def get_eval_score_of_weighted_directions(hidden_states, directions, signs, importances, train_labels, H_mean=None,eval_metric: Union[List[str], str] = "spearmanr"):
    y_score_importance = np.zeros(len(hidden_states))
    # result = {"raw":{}, "importance": {}}
    result = {"raw":{}}
    for i in range(len(directions)):
        y_score_raw = project_onto_direction(hidden_states, directions[i], H_mean=H_mean) * signs[i]
        # y_score_importance += y_score_raw * importances[i]
        result["raw"][i] = get_eval_score_from_prediction(y_score_raw, train_labels, eval_metric)
        # result["importance"][i] = get_eval_score_from_prediction(y_score_importance, train_labels, eval_metric)
    return result

#%%
def get_project_vectors(train_config, validation_config={}):
    # get eval type and metric
    eval_type = train_config.get("eval_type")
    eval_metric = train_config.get("eval_metric")
    if eval_metric is None:
        if eval_type == "pairwise":
            eval_metric = "pair"
        elif eval_type == "absolute":
            eval_metric = "spearmanr"
    # rep config
    token_range = train_config.get("token_range", [-1, 0])
    layer_range = train_config.get("layer_range", [-1, 0])
    
    # setting for get train data
    n_pair = train_config.get("n_pair", 8)
    high_bound = train_config.get("high_bound", 1)
    low_bound = train_config.get("low_bound", 0)
    is_random = train_config.get("is_random", True)
    if eval_type == "pairwise":
        high_bound, low_bound = 1,0
    
    # setting for pca
    n_difference = train_config.get("n_difference", 2)
    n_components = train_config.get("n_components", 4)
    is_recenter = train_config.get("is_recenter", False)
    
    # get train data and validation data
    if validation_config:
        train_representations, train_labels = load_rep_label(train_config)
        validation_representations, validation_labels = load_rep_label(validation_config)
    else:
        representations, labels = load_rep_label(train_config)
        valid_ratio = train_config.get("valid_ratio", 0.1)
        n_sample = len(train_representations)
        if eval_type == "pairwise":
            n_sample = n_sample // 2
        
        # split train and validation data
        seed = train_config.get("seed", 0)
        random.seed(seed)
        n_valid = int(n_sample * valid_ratio)
        valid_index = random.sample(range(n_sample), n_valid)
        train_index = list(set(range(n_sample)) - set(valid_index))
        
        if eval_type == "absolute":
            train_representations = representations[train_index]
            train_labels = labels[train_index]
            validation_representations = representations[valid_index]
            validation_labels = labels[valid_index]
        elif eval_type == "pairwise":
            # train data
            train_representations = []
            train_labels = []
            for i in train_index:
                train_representations.extend([train_representations[i * 2], train_representations[i * 2 + 1]])
                train_labels.extend([labels[i * 2], labels[i * 2 + 1]])
            
            # validation data
            validation_representations = []
            validation_labels = []
            for i in valid_index:
                validation_representations.extend([train_representations[i * 2], train_representations[i * 2 + 1]])
                validation_labels.extend([labels[i * 2], labels[i * 2 + 1]])

    # prepare train emb and labels
    train_emb, train_labels = select_train_data(eval_type, embeddings=train_representations, train_labels=train_labels, high_bound=high_bound, low_bound=low_bound, n_pair=n_pair, is_random=is_random)
    
    # get directions
    pca_model_dict = {}
    for token in range(token_range[0], token_range[1]):
        pca_model_dict[token] = {}
        for layer in tqdm(range(layer_range[0], layer_range[1])):
            hidden_states = train_emb[:, layer, token, :]
            pca_model_dict[token][layer] = get_direction(
                hidden_states=hidden_states, train_labels=train_labels, n_difference=n_difference, n_components=n_components, is_recenter=is_recenter
                )
            
    # get validation result and collect best directions
    best_result = -10000
    best_directions = {
        "layer": None,
        "token": None,
        "directions": None,
        "signs": None,
        "importances": None,
    }
    for layer in tqdm(range(layer_range[0], layer_range[1])):
        for token in range(token_range[0], token_range[1]):
            pca_model = pca_model_dict[token][layer]
            hidden_states = validation_representations[:, layer, token, :]
            validate_result = get_eval_score_of_weighted_directions(
                hidden_states=hidden_states, directions=pca_model[0], signs=pca_model[1], importances=pca_model[2], train_labels=validation_labels, eval_metric=[eval_metric], H_mean=pca_model[3]
                )
            for vector_type in ["raw"]:
                for vector_num in range(n_components):
                    if validate_result[vector_type][vector_num][eval_metric] > best_result:
                        best_result = validate_result[vector_type][vector_num][eval_metric]
                        if vector_type == "raw":
                            best_directions["directions"] = [pca_model[0][vector_num]]
                        best_directions["signs"] = [pca_model[1][vector_num]]
                        best_directions["importances"] = [pca_model[2][vector_num]]
                        best_directions["layer"] = layer
                        best_directions["token"] = token
                        best_directions["vector_type"] = vector_type
                        best_directions["vector_num"] = vector_num
                        best_directions["valid_score"] = validate_result[vector_type][vector_num][eval_metric]
        
    return best_directions

def get_test_result(test_config,directions):
    representations, labels = load_rep_label(test_config)
    layer, token = directions["layer"], directions["token"]
    hidden_states = representations[:, layer, token, :]
    
    eval_type = test_config.get("eval_type")
    eval_metric = test_config.get("eval_metric")
    if eval_metric is None:
        if eval_type == "pairwise":
            eval_metric = ["pair"]
        elif eval_type == "absolute":
            eval_metric = ["spearmanr"]

    test_result = get_eval_score_of_weighted_directions(
        hidden_states=hidden_states, directions=directions["directions"], signs=directions["signs"], importances=directions["importances"], train_labels=labels, eval_metric=eval_metric, H_mean=None
    )
    return test_result

#%%
if __name__ == "__main__":
    train_config = {
        "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "eval_type": "absolute",
        "content_file": "data/example.json",
        "prompt_template": "prompts/hyp_only.txt",
        "rep_label_dir": "./data/representations/example",
        "layer_range": [-32, 0]
    }
    validation_config = train_config
    test_config = {
        "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "eval_type": "absolute",
        "content_file": "data/example.json",
        "prompt_template": "prompts/hyp_only.txt",
        "rep_label_dir": "./data/representations/example",
        "layer_range": [-32, 0]
    }
    directions = get_project_vectors(train_config, validation_config)
    test_result = get_test_result(test_config, directions)

