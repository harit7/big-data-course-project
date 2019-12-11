from utils import *
from GraphPartition import *
from collections import defaultdict
import numpy as np

# random, vertex-cut
def random_vertex_cut(edges, num_parts):
    pmt = np.random.permutation(len(edges))
    ps = int(len(edges) / num_parts)
    lst_parts = []
    for i in range(num_parts):
        pe = [ edges[j] for j in pmt[i * ps : (i + 1) * ps]]
        lst_parts.append(GraphPartition(pe))
    return lst_parts


def greedy_vertex_cut(edges, num_parts):
    pmt = np.random.permutation(len(edges))
    #print(len(pmt))
    lst_parts = [GraphPartition() for i in range(num_parts)]
    A_map = defaultdict(set)
    unassigned_edge_count = defaultdict(int)
    I = np.arange(num_parts)
    
    for e in edges:
        # TODO: directed or undirected??
        u,v = e
        unassigned_edge_count[u]+=1
        unassigned_edge_count[v]+=1
        
    for i in pmt:
        j =0
        u,v = edges[i]
        Au = A_map[u]
        Av = A_map[v]
        s = Au.intersection(Av)
        #print(u,v)
        #print(Au,Av,s)
        if(len(s)>0):
            j = s.pop() # Case 1
            
            #lst_parts[p].add_edge(edges[i])
        else:
            if(len(Au)==0 and len(Av)==0):
                # Case 4
                l = np.array([p.num_edges for p in lst_parts])
                #print(np.argmin(l))
                
                j = get_random_from_set(set(I[l==l[np.argmin(l)]]))

                #
            elif(len(Au) >0 and len(Av)>0):
                # Case 2
                # assign the edge to the partition with vertex (among u,v) with most unassigned edges
                uec_u = unassigned_edge_count[u]
                uec_v = unassigned_edge_count[v]
                if(uec_u>=uec_v):
                    j = get_random_from_set(Au) #Au.pop() 
                else:
                    j = get_random_from_set(Av)#Av.pop()
            else:
                # Au or Av is empty
                if(len(Au)>0):
                    j = get_random_from_set(Au)#Au.pop()
                else:
                    j = get_random_from_set(Av)#Av.pop()
  
        lst_parts[j].add_edge(edges[i])
        unassigned_edge_count[u]-=1
        unassigned_edge_count[v]-=1
        A_map[u].add(j)
        A_map[v].add(j)
        
    return lst_parts

