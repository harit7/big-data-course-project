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

def init_process(rank, size, data_path, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '172.17.180.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, data_path)

def run(rank, size, data_path):
    torch.manual_seed(1234)
    #maintain data of nodes and their allocation
    dict_e2id, lst_vertices = utils.read_e2id(data_path)
    tsr_V = torch.zeros(len(lst_vertices))
    dict_vertex_alloc = utils.read_vertex_alloc_for_parts(data_path)
    
    for i in range(len(lst_vertices)):
        tsr_V[i] = lst_vertices[i]

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
    #test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
 
    #send_id = torch.LongTensor(range(len(tensor_vertices)))
 
    # define the model
    transe = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 10, 
        p_norm = 1, 
        norm_flag = True)


    # define the loss function
    neg_sampling = NegativeSampling(
        model = transe, 
        loss = MarginLoss(margin = 5.0),
        batch_size = train_dataloader.get_batch_size()
    )
    model = transe
    # train the model
    trainer = Trainer(model = neg_sampling, 
                        data_loader = train_dataloader,
                        train_times = 10, alpha = 1.0, 
                        use_gpu = False)
    #trainer.init()
    training_range = tqdm(range(trainer.train_times))
    
    for epoch in training_range:
        res = 0.0
        print('epoch begin')
        
        for data in train_dataloader:
            loss = trainer.train_one_step(data)
            res += loss
        training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
        
        # send recv and update model
        
        #if save_steps and checkpoint_dir and (epoch + 1) % save_steps == 0:
            #print("Epoch %d has finished, saving..." % (epoch))
            #trainer.model.save_checkpoint(os.path.join(checkpoint_dir + "-" + str(epoch) + ".ckpt"))
        #self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        #self.rel_embeddings
    
   
        
        sender_rank = epoch % size
        if(rank == sender_rank):
  
            print('sender', rank)
            receivers = [r for r in range(size) if r != sender_rank]
            for r in receivers:
                # find common vertices or fraction of common vertices to send to r
                print('sending to', r)
                z = torch.zeros(1)
                common_V = dict_vertex_alloc[r]  # TODO: send fraction, randomly or by degree? 
                tsr_common_V = torch.LongTensor(common_V)
                tsr_idx_common_v = torch.LongTensor([ dict_e2id[e] for e in common_V])
                
                z[0] = len(common_V)
                dist.send(tensor=z, dst=r)
                
                ent_embs_send = model.ent_embeddings(tsr_idx_common_v)
                print('sending ', str(z[0]), 'tensors')
                dist.send(tensor=ent_embs_send, dst=r)
                dist.send(tensor=tsr_common_V, dst=r)
                #send relations embedding
                print('sent')
                print(ent_embs_send)
            #ent_embs_recv 
        else:
            print('reciever', rank)
            n = torch.zeros(1)
            Ew = model.ent_embeddings.weight
            Rw = model.rel_embeddings.weight
            #TODO: handle relation embedding sharing
            
            #for r in receivers:
            rcvd = False
            ents_ids_to_update = None
            ent_embs_recv  = None
            multiplier = torch.ones(len(Ew))

            # Send the tensor to process 1
            print('Receiving from ', sender_rank)
            dist.recv(tensor = n, src = sender_rank)
            print('will recv', n, ' embeddings')
            ent_embs_recv = torch.zeros(int(n[0]),transe.dim)
            dist.recv(tensor=ent_embs_recv, src=sender_rank)
            #print(ent_embs_recv[0])
            ents_ids_to_update = torch.LongTensor(int(n[0]))
            dist.recv(tensor=ents_ids_to_update, src=sender_rank)
            
            #print(ents_ids_to_update)
            #print(dict_e2id)
            #print('ents ',ents_ids_to_update)
            #print('ijk ',dict_e2id[int(ents_ids_to_update[0])])

            j = 0
            for e in ents_ids_to_update:
                if( int(e) in dict_e2id):

                    idx = dict_e2id[int(e)]
                    #print(idx,e)
                    Ew.data[idx]+= ent_embs_recv[j]
                    multiplier[idx]+=1
                j+=1

            # recieved from all, now reduce
            print(multiplier[:15])
            for i in range(len(Ew)):
                if(multiplier[i]>1):
                    Ew.data[i] = Ew.data[i]*(1.0/multiplier[i])


    # test the model
    #transe.load_checkpoint('./checkpoint/transe.ckpt')
    #tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
    #tester.run_link_prediction(type_constrain = False)

if __name__ == "__main__":
    size = int(sys.argv[2])
    rank = int(sys.argv[1])
    #print(size,rank)
    data_dir = './sbm_n_1000_parts_3_p1_0.01_p2_0.001_num_edges_3689/' 
    #data_dir += 'gvc_part_'+str(rank)+'_3/'
    data_dir += 'gvc_part_' + str(rank) + '_' + size + '/'
    print(data_dir)
    #run(rank,size,data_dir)
    #edge_file_path = 'gvc_part_'+str(rank)+'_3_sbm_n_1000_parts_3_p1_0.01_p2_0.001_num_edges_3689_.edgelist'
    p = Process(target=init_process, args=(rank, size, data_dir , run))
    p.start()
    p.join()
