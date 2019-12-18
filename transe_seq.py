import os
import torch

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

def run(rank, size,data_path):
    torch.manual_seed(1234)

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path = data_path, 
        nbatches = 100,
        threads = 1, 
        sampling_mode = "normal", 
        bern_flag = 1, 
        filter_flag = 1, 
        neg_ent = 25,
        neg_rel = 0)

    # dataloader for test
    test_dataloader = TestDataLoader(data_path, "link")

    # define the model
    transe = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 25, 
        p_norm = 2, 
        norm_flag = True)


    # define the loss function
    model = NegativeSampling(
        model = transe, 
        loss = MarginLoss(margin = 5.0),
        batch_size = train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 90, alpha = 1.0, use_gpu = False)
    trainer.run()
    transe.save_checkpoint('./checkpoint/transe.ckpt')

    # test the model
    transe.load_checkpoint('./checkpoint/transe.ckpt')
    tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = False)
    tester.run_link_prediction(type_constrain = False)
    print('random model')
    transe_rand = TransE(
        ent_tot = train_dataloader.get_ent_tot(),#train_dataloader.get_ent_tot(),
        rel_tot = 1,# train_dataloader.get_rel_tot(),
        dim = 25, 
        p_norm = 2, 
        norm_flag = True)    
       # test the model
 
    tester = Tester(model = transe_rand, data_loader = test_dataloader, use_gpu = False)
    tester.run_link_prediction(type_constrain = False)

if __name__ == "__main__":
    #size = int(sys.argv[1])
    #rank = int(sys.argv[1])
    #print(size,rank)
    data_dir = './sbm_n_1000_parts_3_p1_0.01_p2_0.001_num_edges_3689/' 
    #data_dir += 'gvc_part_'+str(rank)+'_3/'
    print(data_dir)
    run(0,1,data_dir)
    #edge_file_path = 'gvc_part_'+str(rank)+'_3_sbm_n_1000_parts_3_p1_0.01_p2_0.001_num_edges_3689_.edgelist'
    #p = Process(target=init_process, args=(rank, size, edge_file_path, run))
    #p.start()
    #p.join()
