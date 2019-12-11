from utils import *
from GraphPartition import *
from collections import defaultdict
import random


def random_edge_cut(edges, num_parts):
    #evenly distribute vertices 
    #? replicate the edges cut?? or discard the edge cut?? 
    # what to do with the edges cut?? 
    vertex_edges = defaultdict(list)
    for edge in edges:
        vertex_edges[edge[0]].append(tuple(edge))
        vertex_edges[edge[1]].append((edge[1], edge[0]))
    
    partition_set = defaultdict(set)
    for k, v in vertex_edges.items():
        select_part = random.randint(0, num_parts - 1)
        for edge in v:
            partition_set[select_part].add(edge)

    lst_parts = []
    for k, v in partition_set.items():
        lst_parts.append(GraphPartition(list(v)))
    
    return lst_parts    


def greedy_edge_cut(edges,num_parts):
    # apply edge cut first.
    # cut the vertex with high degree??
    pass


if __name__ == "__main__":
    edges = []
    edges.append([1, 2])
    edges.append([2, 3])
    edges.append([4, 5])
    edges.append([5, 6])
    edges.append([3, 5])
    result = random_edge_cut(edges, 3)
    for part in result:
        print(part)
