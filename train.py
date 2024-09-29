import torch

from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_label_loss, new_triplet_loss,new_triplet_local_loss
from utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os
import pandas as pd
import gc

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
# torch.cuda.set_per_process_memory_fraction(device=device, fraction=0.8)

type = "test_func"

dataset_name = "glpk_arhee"

# device = torch.device('cpu')
# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))


model_path = config['model_dir']

# basic_path = "/home/ubuntu/Desktop/gmm/Graph-Matching-Networks/GMN/train_dataset/dataset_oobert/sqlite3_arch/result2"
# auc_path =  basic_path + os.sep + "auc.csv"
# result_path = basic_path + os.sep + "siml.csv"
# loss_path = basic_path + os.sep + "loss.csv"

print (model_path)
print (config['training'])

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

training_set, validation_set = new_build_datasets(config)

# if config['training']['mode'] == 'pair':
#     training_data_iter = training_set.pairs()
#     first_batch_graphs, _ = next(training_data_iter)
# else:
#     training_data_iter = training_set.triplets()
#     first_batch_graphs = next(training_data_iter)

node_feature_dim = 768
edge_feature_dim = 1536

# node_feature_dim = first_batch_graphs.node_features.shape[-1]
# edge_feature_dim = first_batch_graphs.edge_features.shape[-1]
print("node_feature_dim:" + str(node_feature_dim))
print("edge_feature_dim:" + str(edge_feature_dim))

model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

accumulated_metrics = collections.defaultdict(list)

training_n_graphs_in_batch = config['training']['batch_size']
if config['training']['mode'] == 'pair':
    training_n_graphs_in_batch *= 2
elif config['training']['mode'] == 'triplet':
    training_n_graphs_in_batch *= 2
else:
    raise ValueError('Unknown training mode: %s' % config['training']['mode'])


if os.path.exists(model_path):

    if type == "test_func":

        model = torch.load(config['model_dir'])

        inputs_list = os.listdir(config['testing']['functions_list_inputs'])

        for func_name in inputs_list:
            print(func_name)
            df_input_path = config['testing']['functions_list_inputs'] + os.sep + func_name
            df_output_path = config['testing']['functions_list_outputs'] + os.sep + func_name

            df = pd.read_csv(df_input_path, index_col=0)

            batch_generator = build_testing_generator(
                config,
                df_input_path)

            similarity_list = list()
            model.eval()
            with torch.no_grad():
                accumulated_pair_auc = []
                accumulated_pair_sim = []
                accumulated_pair_label = []

                for batch in batch_generator.pairs():

                    node_features, edge_features, from_idx, to_idx, graph_idx, from_idx_list, to_idx_list, graph_ad, edge_idx = get_graph(
                        batch)

                    # eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                    #                    to_idx.to(device),
                    #                    graph_idx.to(device), config['evaluation']['batch_size'] * 2)

                    eval_pairs, node_status = model(node_features.to(device), edge_features.to(device),
                                                    from_idx.to(device),
                                                    to_idx.to(device),
                                                    graph_idx.to(device), config['evaluation']['batch_size'] * 2,
                                                    from_idx_list.to(device), to_idx_list.to(device),
                                                    graph_ad.to(device), edge_idx.to(device))

                    x, y = reshape_and_split_tensor(eval_pairs, 2)
                    similarity = compute_similarity(config, x, y)
                    # pair_auc = auc(similarity, labels)
                    # accumulated_pair_auc.append(pair_auc)

                    simls = similarity.cpu().detach().numpy().tolist()
                    for siml in simls:
                        accumulated_pair_sim.append(siml)

            df['sim'] = accumulated_pair_sim[:df.shape[0]]

            df.to_csv(df_output_path)
            print("Result CSV saved to {}".format(df_output_path))

            gc.collect()
            torch.cuda.empty_cache()

