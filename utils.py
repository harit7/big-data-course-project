import random
import shutil
import os

def read_edge_list(data_dir):
    edges = []
    f = open(data_dir+'/orig_triples.txt')
    n = int(f.readline())
    for l in f:
        s = l.rstrip('\n').split('\t')
        edges.append([int(s[0]), int(s[1])])
    f.close()    
    return edges

def write_edge_list(edges,file_path):
    f = open(file_path,'w')
    for e in edges:
        f.write(str(e[0])+" "+str(e[1])+"\n")
    f.flush()
    f.close()

def read_e2id(data_dir):
    e2id = dict()
    f = open(data_dir+'/entity2id.txt')
    n = f.readline()
    for l in f:
        l = l.rstrip('\n').split('\t')
        e2id[int(l[0])] = int(l[1])
    f.close()

def write_as_triples(edges,d):
    ents = set()
    r = 0
    for e in edges:
        ents.add(e[0])
        ents.add(e[1])

    d_ents = dict(zip(sorted(list(ents)),range(len(ents))))
    f1 = open(d+'/entity2id.txt','w')
    f1.write(str(len(ents))+"\n")
    for k in d_ents.keys():
        f1.write(str(k)+'\t'+str(d_ents[k])+'\n')
    f1.flush()
    f1.close()
    
    f2 = open(d+'/orig_triples.txt','w')
    f3 = open(d+'/train2id.txt','w')
    ents = set()
    r = 0
    f2.write(str(len(edges))+"\n")    
    f3.write(str(len(edges))+"\n")
    for e in edges:
        f2.write(str(e[0]) +'\t'+str(e[1])+"\t"+str(r) +"\n")
        f3.write( str(d_ents[e[0]]) +'\t'+str(d_ents[e[1]])+ '\t'+'0'+'\n')
    f2.flush()
    f3.flush()
    f2.close()
    f3.close()
    
    f4 = open(d+'/relation2id.txt','w')
    f4.write('1\n')
    f4.write(str(r)+'\t'+'0')
    f4.flush()
    f4.close()
    
def read_e2id(data_path):
    lst_vertices = []
    dict_e2id  = {}
    e2id_f = open(data_path+'/entity2id.txt')
    e2id_f.readline()
    for l in e2id_f:
        s = l.rstrip('\n').split('\t')
        e = int(s[0])
        _id = int(s[1])
        dict_e2id[e] = _id
        lst_vertices.append(e)
    e2id_f.close()
    return dict_e2id, lst_vertices

def clean_mkdir(d):
    if os.path.exists(d):
        print('directory exists, removing it')
        shutil.rmtree(d)
    os.mkdir(d)
    
def get_random_from_set(s):
    i = 0
    if(len(s)==1):
        i = 0
    else:
        i = random.randint(0,len(s)-1)
    return list(s)[i]