class GraphPartition:
    def __init__(self, edges=None):
        self.vertices = set()
        if(edges is None):
            self.num_edges = 0
            self.num_vertices = 0
            self.edges = []
        else:   
            self.num_edges = len(edges)
            self.edges = edges

            for e in edges:
                self.vertices.add(e[0])
                self.vertices.add(e[1])

            self.num_vertices = len(self.vertices)
        
    def to_adj_lst(self):
        pass
    
    def get_vertex_set(self):
        return self.vertices
    
    def add_edge(self, e):
        self.edges.append(e)
        self.vertices.add(e[0])
        self.vertices.add(e[1])
        self.num_vertices = len(self.vertices)
        self.num_edges+=1
    
    def num_common_vertices(self, p2):
        return len(self.vertices.intersection(p2.vertices))