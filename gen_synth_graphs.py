import networkx as nx
import numpy as np
import os
from utils import *

def sbm(num_nodes, num_parts, intra_block_prob, inter_block_prob):
    N = num_nodes
    p = num_parts
    p1 = intra_block_prob
    p2 = inter_block_prob
    sizes = [int(N/p)]*p
    P = np.array([[p2/(p-1)]*p]*p)

    for i in range(p):
        P[i][i] = p1

    g = nx.stochastic_block_model(sizes, P, nodelist=None, seed=0, directed=True, selfloops=False, sparse=True)
    return g
    #dgs = [d for k,d in g.degree()]

def power_law():
    pass

p1 = 0.01
p2 = 0.001
N  = 1000
parts = 3
g  = sbm(N,3,p1,p2)
key = 'sbm_'+'n_'+str(N)+'_parts_'+str(parts)+'_p1_'+str(p1) +'_p2_'+str(p2)+'_num_edges_'+str(g.size())
print(key)
clean_mkdir(key)

write_as_triples(g.edges(),key)

#nx.write_edgelist(g,key+'full.edgelist',data=False)
