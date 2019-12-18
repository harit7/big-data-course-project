import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from torch.random import *
import torch.optim as optim
from math import ceil
    
import torch.nn as nn
import torch.nn.functional as F
import sys
from random import Random
import torch.optim as optim
sys.path.append('./OpenKE')
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from tqdm import tqdm
import utils

data_dir = './sbm_n_1000_parts_3_p1_0.01_p2_0.001_num_edges_3689/' 
part_type = 'rec'
full_dict_e2id, full_lst_vertices = utils.read_e2id(data_dir)
transe_full = TransE(
        ent_tot = len(full_lst_vertices),#train_dataloader.get_ent_tot(),
        rel_tot = 1,# train_dataloader.get_rel_tot(),
        dim = 25, 
        p_norm = 2, 
        norm_flag = True)
transe_full.ent_embeddings.weight.data = torch.zeros((len(full_lst_vertices),25))
Z = transe_full.ent_embeddings.weight.data
C = torch.zeros(len(full_lst_vertices))
size = 3
for rank in range(size):
    part_data_dir = data_dir+part_type+'_part_' + str(rank) + '_' + str(size)+'/' 
    part_dict_e2id, part_lst_vertices = utils.read_e2id(part_data_dir)

    transe_part = TransE(
            ent_tot = len(part_lst_vertices),#train_dataloader.get_ent_tot(),
            rel_tot = 1,# train_dataloader.get_rel_tot(),
            dim = 25, 
            p_norm = 2, 
            norm_flag = True)

    transe_part.load_checkpoint(part_data_dir+'/checkpoint/transe.ckpt')
    print(len(part_lst_vertices))
    for i in range(len(part_lst_vertices)):
        
        e = part_lst_vertices[i]
        ei = full_dict_e2id[e]
        #print(i,e,ei)
        Z[ei]+=transe_part.ent_embeddings.weight[i]
        C[ei]+=1
    #print(C)
    
    
    # dataloader for training
test_dataloader = TestDataLoader(data_dir, "link")
tester = Tester(model = transe_full, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)
