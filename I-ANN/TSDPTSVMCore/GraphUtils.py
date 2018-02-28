import networkx as nx

"""
original networkx algorithm: 
def all_paths(G, source, target, cutoff=None):
    if cutoff == None:
        cutoff = len(G)-1

    if cutoff < 1:
        return

    visited = [source]
    stack = [(v for u,v in G.edges(source))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        elif len(visited) < cutoff:
            if child == target:
                yield visited + [target]
            elif child not in visited:
                visited.append(child)
                stack.append((v for u,v in G.edges(child)))
        else: #len(visited) == cutoff:
            count = ([child]+list(children)).count(target)
            for i in range(count):
                yield visited + [target]
            stack.pop()
            visited.pop()
"""

class X:
    def __init__(self, _x , _data):
        self.x = _x
        self.data = _data

    def delx(self):
        del self.x
        del self.data
    
    def get_all(self):
        return (self.x , self.data)
    
    
def LOWLEVEL_all_paths(G, source, target, cutoff=None):
    if cutoff == None:
        cutoff = len(G)-1
        
    if cutoff < 1:
        return
    
    if (not G.has_node(source)) or (not G.has_node(target)) or (source == target):
        return 
    
    source = X(source,None)
    target = X(target,None)
    
    visited = [source]
    
    stack = [( X(v,(u,v,d)) for u,v,d in G.edges(source.x,data=True))]
   
    while stack:
        children = stack[-1]
        child = next(children, None)
        
        if child is None:
            stack.pop()
            visited.pop()

        elif len(visited) < cutoff:
            if child.x == target.x:
                yield visited + [X(target.x, child.data)]

            elif child.x not in [i.x for i in visited]:
                visited.append(child)
                stack.append(  (X(v,(u,v,d)) for u,v,d in G.edges(child.x,data=True))   )

        else: #len(visited) == cutoff:
            count = ([child.x]+list(ch.x for ch in children)).count(target.x)
            """
            for i in range(count):
                yield visited + [X (target.x, None)]
            """
            if count:
                this_node_id = visited[-1].x
                target_node_id = target.x 
                final_edges = (X(target_node_id,(this_node_id,target_node_id,d)) for d in G[this_node_id][target_node_id].itervalues())
                for i in range(count):
                    #print "YIEEEEEELD"
                    yield visited + [next(final_edges,None)]
            stack.pop()
            visited.pop()

def LOWLEVEL_all_paths_withweights (G, source, target, edge_weight_name="weight", path_length_cutoff=None):
    for path in LOWLEVEL_all_paths(G, source, target, path_length_cutoff):
        weight = sum([i.get_all()[1][2][edge_weight_name] for i in path if i.get_all()[1] is not None])
        yield (weight,path)

def get_all_topK_paths_withweights (G, source, target, edge_weight_name="weight", K=None, path_length_cutoff=None):
    import sys ; 

    #if K==None: --> return all paths , else, return top K paths
    if K is not None:
        if (not isinstance(K,int)) or (K < 1):
            print "K should be integer and at least >= 1 or None" 
            sys.exit (-1)
            
    #generator is NOW converted into list and then sorted 
    #sorting based on path_weight, ascending ... 
    all_paths = sorted(LOWLEVEL_all_paths_withweights (G, source, target, edge_weight_name, path_length_cutoff), key=lambda x: x[0])
    #print len(all_paths)
       
    if K is not None:
        if K < len(all_paths): #there are X paths, we need only first K of them 
            tmp_paths = all_paths[0:K]
            max_weight = tmp_paths[-1][0]
            
            # if weights = [2,3,5,5,6,7,11,15,34] and K = 3 ---> [2,3,5,5] because 
            for X in all_paths[K:]:
                this_weight = X[0]
                if this_weight == max_weight:
                    tmp_paths.append(X)
                else:
                    break ; 
            all_paths = tmp_paths
    
    results = [] 
    ALL_NODES_IDS = set()
    ALL_NODES_INFO = {}
    
    for weight , path_info in all_paths:
        ALL_NODES_IDS = ALL_NODES_IDS.union (set([i.x for i in path_info]))
        results.append ( (weight, [i.data for i in path_info if i.data is not None]))
    
    for node_id in ALL_NODES_IDS:
        ALL_NODES_INFO[node_id] = G.node[node_id]
    
    return [results , ALL_NODES_INFO] #returns [] if no SDP

def get_all_topK_paths_withweights_RAMFRIENDLY (G, source, target, edge_weight_name="weight", K=None, path_length_cutoff=None):
    import sys ; 

    #if K==None: --> return all paths , else, return top K paths
    if K is not None:
        if (not isinstance(K,int)) or (K < 1):
            print "K should be integer and at least >= 1 or None" 
            sys.exit (-1)
            
    #generator is NOW converted into list and then sorted 
    #sorting based on path_weight, ascending ... 
    all_paths = [] 
    counter = 0 
    MAX_PATHS_IN_RAM = 20000
    for p in LOWLEVEL_all_paths_withweights (G, source, target, edge_weight_name, path_length_cutoff):
        counter += 1
        all_paths.append (p)
        
        if (counter == MAX_PATHS_IN_RAM):
            counter = 0             
            if K is not None:
                all_paths = sorted(all_paths, key=lambda x: x[0])    
                if K < len(all_paths): #there are X paths, we need only first K of them 
                    tmp_paths = all_paths[0:K]
                    max_weight = tmp_paths[-1][0]
                    # if weights = [2,3,5,5,6,7,11,15,34] and K = 3 ---> [2,3,5,5] because 
                    for X in all_paths[K:]:
                        this_weight = X[0]
                        if this_weight == max_weight:
                            tmp_paths.append(X)
                        else:
                            break ; 
                    all_paths = tmp_paths
    
    
    all_paths = sorted(all_paths, key=lambda x: x[0])    
    if K is not None:
        if K < len(all_paths): #there are X paths, we need only first K of them 
            tmp_paths = all_paths[0:K]
            max_weight = tmp_paths[-1][0]
            
            # if weights = [2,3,5,5,6,7,11,15,34] and K = 3 ---> [2,3,5,5] because 
            for X in all_paths[K:]:
                this_weight = X[0]
                if this_weight == max_weight:
                    tmp_paths.append(X)
                else:
                    break ; 
            all_paths = tmp_paths
    
    results = [] 
    ALL_NODES_IDS = set()
    ALL_NODES_INFO = {}
    
    for weight , path_info in all_paths:
        ALL_NODES_IDS = ALL_NODES_IDS.union (set([i.x for i in path_info]))
        results.append ( (weight, [i.data for i in path_info if i.data is not None]))
    
    for node_id in ALL_NODES_IDS:
        ALL_NODES_INFO[node_id] = G.node[node_id]
    
    return [results , ALL_NODES_INFO] #returns [] if no SDP

  
#example: 
"""
if __name__ == "__main__":
    G = nx.MultiGraph()
    G.add_nodes_from ([ (1, {"txt":"a"}) , 
                        (2, {"txt":"b"}) , 
                        (3, {"txt":"c"}) , 
                        (4, {"txt":"d"}) , 
                        (5, {"txt":"e"}) , 
                        (6, {"txt":"f"}) , 
                        (7, {"txt":"g"}) ,
                        (8, {"txt":"h"}) ,

])
                        
    G.add_edge(1,2,weight=1,tp="12")
    
    G.add_edge(2,3,weight=1,tp="23_1")
    G.add_edge(2,3,weight=2,tp="23_2")
    
    G.add_edge(3,4,weight=1,tp="34")
    G.add_edge(4,5,weight=1,tp="45")
    
    G.add_edge(5,6,weight=1,tp="56_1")
    G.add_edge(5,6,weight=2,tp="56_2")
    
    G.add_edge(1,5,weight=8,tp="15")
    G.add_edge(1,6,weight=16,tp="16")
    G.add_edge(6,6,weight=16,tp="66")
    G.add_edge(1,7,weight=16,tp="17")
    #all_paths1 = get_all_topK_paths_withweights (G, 1,6, K=23,path_length_cutoff= 5)
    all_paths2 = get_all_topK_paths_withweights_RAMFRIENDLY (G, 1,6, K=23,path_length_cutoff= 5)
    
 
    for i,j in all_paths2[0]:
        print i,j
        print "-"*40
"""