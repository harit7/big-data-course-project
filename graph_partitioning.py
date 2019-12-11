import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from utils import *
from vertex_cuts import *
from edge_cuts import *
from collections import defaultdict


np.random.seed(0)

# assume the graph fits in memory of a machine.
#data_dir = 'sbm_n_1000_parts_3_p1_0.01_p2_0.001_num_edges_3689'
data_dir = 'power_law_n_100_parts_3_num_edge_290'
#fname = 'sbm_n_1000_parts_3_p1_0.01_p2_0.001_num_edges_3689_.edgelist'
lst_edges = read_edge_list(data_dir)
num_parts = 3
lst_parts = random_edge_cut(lst_edges, num_parts)
for i in range(num_parts):
    print(lst_parts[i].num_edges)
    for j in range(i+1):
        print(i, j, lst_parts[i].num_vertices,lst_parts[i].num_common_vertices(lst_parts[j]))

for i in range(num_parts):
    part_dir = data_dir+'/gvc_part_'+str(i)+'_'+str(num_parts)
    clean_mkdir(part_dir)
    write_as_triples(lst_parts[i].edges,part_dir)
    write_vertex_alloc_for_parts(i,lst_parts,part_dir)
    #write_edge_list(lst_parts[i].edges,'gvc_part_'+str(i)+"_"+str(num_parts)+"_"+fname)