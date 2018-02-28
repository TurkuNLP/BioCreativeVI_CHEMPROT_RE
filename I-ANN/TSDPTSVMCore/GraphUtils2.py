import networkx as nx
from MAXHEAP import CONTROLLED_MAX_HEAP 
from MAXHEAP import CONTROLLED_MAX_CONTRAINER

# (1) ORIGINAL algorithm BY NETWORKX DEVELOPERS, 
#     DOES NOT RETURN EDGES, JUST NODES FOR MULTIGRAPH
#     One version with path_length cutoff
#     One version without path_length cutoff
def ORIGINAL_all_simple_paths_multigraph_NOCUTOFF(G, source, target):
    visited = [source]
    stack = [(v for u,v in G.edges(source))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        else:
            if child == target:
                yield visited + [target]
            elif child not in visited:
                visited.append(child)
                stack.append((v for u,v in G.edges(child)))

def ORIGINAL_all_simple_paths_multigraph_CUTOFF(G, source, target, cutoff=None):
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
            
############################################################################################################
# (2) MY IMPLEMENTATIONs: yields/returns edges, with/without cutoff
def _all_simple_paths_multigraph_NOCUTOFF_YIELDEDGES(G, source, target):
    #Having edge:
    #    - edge[0] --> edge_source
    #    - edge[1] --> edge_target
    #    - edge[2] --> edge_data
    visited = []
    stack  = [ (i for i in G.edges(source,data=True) if i[1]<> source) ] #list with one element, which is generator
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None)        
        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        else:
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                yield visited + [next_edge]
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                visited.append (next_edge)
                stack.append ( (i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]) )

def _all_simple_paths_multigraph_NOCUTOFF_RETEDGES(G, source, target):
    results = []
    visited = []
    stack  = [ (i for i in G.edges(source,data=True) if i[1]<> source) ] #list with one element, which is generator
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None)        
        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        else:
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                results.append (visited + [next_edge])
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                visited.append (next_edge)
                stack.append ( (i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]) )
    return results 

def _all_simple_paths_multigraph_CUTOFF_YIELDEDGES(G, source, target,cutoff=None):
    if cutoff == None:
        cutoff = len(G)-1
    if cutoff < 1:
        return
    visited = []
    stack   = [ (i for i in G.edges(source,data=True) if i[1]<> source) ] #list with one element, which is generator
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None)        
        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        
        elif len(visited) < cutoff-1: #<<<CRITICAL>>> MINUS ONE IS NEEDED, CAUSE WE ARE VISITING EDGES (e.x: a-b-c ==> 2 edges)
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                yield visited + [next_edge]
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                visited.append (next_edge)
                stack.append ( (i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]) )

        else: #len(visited) == cutoff
            for edge in [next_edge]+list(all_outgoing_edges):
                edge_source , edge_target , edge_data = edge
                if edge_target == target:
                    yield visited + [edge]
            try:
                stack.pop()
                visited.pop()
            except:
                pass 

def _all_simple_paths_multigraph_CUTOFF_YIELDEDGES_AND_WEIGHTS(G, source, target,cutoff=None):
    if cutoff == None:
        cutoff = len(G)-1
    if cutoff < 1:
        return
    visited = []
    stack   = [ (i for i in G.edges(source,data=True) if i[1]<> source) ] #list with one element, which is generator
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None)        
        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        
        elif len(visited) < cutoff-1: #<<<CRITICAL>>> MINUS ONE IS NEEDED, CAUSE WE ARE VISITING EDGES (e.x: a-b-c ==> 2 edges)
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                p = visited + [next_edge]
                w = sum(i[2]["w"] for i in p)
                yield (w,p)
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                visited.append (next_edge)
                stack.append ( (i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]) )

        else: #len(visited) == cutoff
            for edge in [next_edge]+list(all_outgoing_edges):
                edge_source , edge_target , edge_data = edge
                if edge_target == target:
                    p = visited + [edge]
                    w = sum(i[2]["w"] for i in p)
                    yield (w,p)
            try:
                stack.pop()
                visited.pop()
            except:
                pass 

##############################################################################################################
############################################## TOP K PATHS ###################################################
def _all_TopKPaths_ControlledMaxHeap (G, source, target, KTop, cutoff=None):
    if cutoff == None:
        cutoff = len(G)-1
    if cutoff < 1:
        return []
    TopKPaths = CONTROLLED_MAX_HEAP(KTop)
    visited = []
    stack   = [ (i for i in G.edges(source,data=True) if i[1]<> source) ] #list with one element, which is generator
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None)        
        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        
        elif len(visited) < cutoff-1: #<<<CRITICAL>>> MINUS ONE IS NEEDED, CAUSE WE ARE VISITING EDGES (e.x: a-b-c ==> 2 edges)
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                p = visited + [next_edge]
                w = sum(i[2]["w"] for i in p)
                TopKPaths.push ((w,p))    
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                visited.append (next_edge)
                stack.append ( (i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]) )

        else: #len(visited) == cutoff
            for edge in [next_edge]+list(all_outgoing_edges):
                edge_source , edge_target , edge_data = edge
                if edge_target == target:
                    p = visited + [edge]
                    w = sum(i[2]["w"] for i in p)
                    TopKPaths.push ((w,p))    
            try:
                stack.pop()
                visited.pop()
            except:
                pass 

    return TopKPaths.get_all()

def _all_TopKPaths (G, source, target, K=None, cutoff=None):
    all_paths = [] 
    counter = 0 
    MAX_PATHS_IN_RAM = 20000
    for p in _all_simple_paths_multigraph_CUTOFF_YIELDEDGES_AND_WEIGHTS (G, source, target, cutoff):
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
    
    return all_paths

######################################################### TOPK_PATHS WITH HEURISTICS #########################################
def _all_TopKPaths_ControlledMaxHeap_Heuristic (G, source, target, KTop, cutoff=None):
    if cutoff == None:
        cutoff = len(G)-1
    if cutoff < 1:
        return []
    SPLTT = nx.shortest_path_length (G,source=None,target=target, weight="w") #shortest path lengths to THE target 
    TopKPaths = CONTROLLED_MAX_HEAP(KTop)
    visited = []
    stack  = [ (b for b in sorted ((i for i in G.edges(source,data=True) if i[1]<> source), key=lambda x:SPLTT[x[1]]+x[2]["w"])) ] #sort based on shortest distance to target
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None) 

        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        
        elif len(visited) < cutoff-1: #<<<CRITICAL>>> MINUS ONE IS NEEDED, CAUSE WE ARE VISITING EDGES (e.x: a-b-c ==> 2 edges)
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                p = visited + [next_edge]
                w = sum ([i[2]["w"] for i in p])
                TopKPaths.push ((w,p))    
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                #check if it is promissing ...
                sofar_path_weight = sum ([i[2]["w"] for i in visited]) + edge_data["w"]
                best_weight_to_destination = sofar_path_weight + SPLTT[edge_target]
                if best_weight_to_destination < TopKPaths.max(): #critical < not <=, because max_heap will not hold more than x-top, so there is no point to update with something else with the same weight
                    visited.append (next_edge)
                    stack.append ( (z for z in sorted ((i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]), key=lambda x:SPLTT[x[1]])) )

        else: #len(visited) == cutoff
            for edge in [next_edge]+list(all_outgoing_edges):
                edge_source , edge_target , edge_data = edge
                if edge_target == target:
                    p = visited + [edge]
                    w = sum(i[2]["w"] for i in p)
                    TopKPaths.push ((w,p))    
            try:
                stack.pop()
                visited.pop()
            except:
                pass 

    return sorted(TopKPaths.get_all(), key=lambda x: (x[0],len(x[1]))) #first sort based on weight, then based on number of edges in the path

def _all_TopKPaths_ControlledMaxContainer_Heuristic (G, source, target, KTop, cutoff=None):
    if cutoff == None:
        cutoff = len(G)-1
    if cutoff < 1:
        return []
    SPLTT = nx.shortest_path_length (G,source=None,target=target, weight="w") #shortest path lengths to THE target 
    TopKPaths = CONTROLLED_MAX_CONTRAINER(KTop)
    visited = []
    stack  = [ (b for b in sorted ((i for i in G.edges(source,data=True) if i[1]<> source), key=lambda x:SPLTT[x[1]]+x[2]["w"])) ] #sort based on shortest distance to target
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None) 

        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        
        elif len(visited) < cutoff-1: #<<<CRITICAL>>> MINUS ONE IS NEEDED, CAUSE WE ARE VISITING EDGES (e.x: a-b-c ==> 2 edges)
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                p = visited + [next_edge]
                w = sum ([i[2]["w"] for i in p])
                TopKPaths.push ((w,p))    
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                #check if it is promissing ...
                sofar_path_weight = sum ([i[2]["w"] for i in visited]) + edge_data["w"]
                best_weight_to_destination = sofar_path_weight + SPLTT[edge_target]
                if best_weight_to_destination <= TopKPaths.max(): #critical <= and not <
                    visited.append (next_edge)
                    stack.append ( (z for z in sorted ((i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]), key=lambda x:SPLTT[x[1]])) )

        else: #len(visited) == cutoff
            for edge in [next_edge]+list(all_outgoing_edges):
                edge_source , edge_target , edge_data = edge
                if edge_target == target:
                    p = visited + [edge]
                    w = sum(i[2]["w"] for i in p)
                    TopKPaths.push ((w,p))    
            try:
                stack.pop()
                visited.pop()
            except:
                pass 

    return sorted(TopKPaths.get_all(), key=lambda x: (x[0],len(x[1]))) #first sort based on weight, then based on number of edges in the path

def _all_TopKPaths_ControlledMaxContainer_Heuristic_GuarantySP (G, source, target, KTop, cutoff=None):
    #this one guaranties that REAL shortest path is always calculated and added and returned regardless of cutoff
    if cutoff == None:
        cutoff = len(G)-1
    if cutoff < 1:
        return []
    TopKPaths = CONTROLLED_MAX_CONTRAINER(KTop)

    for shortest_path in all_shortest_paths_edges(G,source, target):
        shortest_path_weight = sum ([i[2]["w"] for i in shortest_path])
        TopKPaths.push ((shortest_path_weight,shortest_path))
        
    SPLTT = nx.shortest_path_length (G,source=None,target=target, weight="w") #shortest path lengths to THE target 
    visited = []
    stack  = [ (b for b in sorted ((i for i in G.edges(source,data=True) if i[1]<> source), key=lambda x:SPLTT[x[1]]+x[2]["w"])) ] #sort based on shortest distance to target
    while stack:
        all_outgoing_edges = stack[-1]
        next_edge = next(all_outgoing_edges,None) 

        if next_edge is None:
            stack.pop()
            try:            
                visited.pop() 
            except:
                pass
        
        elif len(visited) < cutoff-1: #<<<CRITICAL>>> MINUS ONE IS NEEDED, CAUSE WE ARE VISITING EDGES (e.x: a-b-c ==> 2 edges)
            edge_source , edge_target , edge_data = next_edge
            
            if edge_target == target:
                p = visited + [next_edge]
                w = sum ([i[2]["w"] for i in p])
                TopKPaths.push ((w,p))    
            
            elif edge_target not in [i[0] for i in visited]+[i[1] for i in visited]:
                #check if it is promissing ...
                sofar_path_weight = sum ([i[2]["w"] for i in visited]) + edge_data["w"]
                best_weight_to_destination = sofar_path_weight + SPLTT[edge_target]
                if best_weight_to_destination <= TopKPaths.max(): #critical <= and not <
                    visited.append (next_edge)
                    stack.append ( (z for z in sorted ((i for i in G.edges(edge_target,data=True) if i[1] not in [b[0] for b in visited]+[b[1] for b in visited]), key=lambda x:SPLTT[x[1]])) )

        else: #len(visited) == cutoff
            for edge in [next_edge]+list(all_outgoing_edges):
                edge_source , edge_target , edge_data = edge
                if edge_target == target:
                    p = visited + [edge]
                    w = sum(i[2]["w"] for i in p)
                    TopKPaths.push ((w,p))    
            try:
                stack.pop()
                visited.pop()
            except:
                pass 

    return sorted(TopKPaths.get_all(), key=lambda x: (x[0],len(x[1]))) #first sort based on weight, then based on number of edges in the path

def _all_TopKPaths_Info (G, source, target, KTop, cutoff):
    #results  = _all_TopKPaths (G, source, target, KTop, cutoff) #normal, ramfriendly, sorting, no max heap, no heuristic
    #results = _all_TopKPaths_ControlledMaxHeap (G, source, target, KTop, cutoff) #using max-heap, no sorting, no heuristic
    #results = _all_TopKPaths_ControlledMaxHeap_Heuristic (G, source, target, KTop, cutoff) #using max-heap AND heuristic
    #results = _all_TopKPaths_ControlledMaxContainer_Heuristic(G, source, target, KTop, cutoff) #using max-container (sorting always) AND heuristic
    results = _all_TopKPaths_ControlledMaxContainer_Heuristic_GuarantySP (G, source, target, KTop, cutoff) #same as above, but guaranties that always all SDPs are found regardless of cutoff! 
    ALL_NODES_IDS , ALL_NODES_INFO = [] , {}
    for weight , path in results:
        ALL_NODES_IDS.extend ( [edge[0] for edge in path] + [edge[1] for edge in path] )
    for node_id in set(ALL_NODES_IDS):
        ALL_NODES_INFO[node_id] = G.node[node_id]
    return [results , ALL_NODES_INFO] 

def all_shortest_paths_edges (G, source, target):
    import copy
    paths = set(tuple(i) for i in nx.all_shortest_paths (G,source,target,weight="w"))
    final_results = []

    for path in paths:
        this_path_all_combinations = [] 
        
        for i in range(len(path)-1):
            this_node = path[i]
            next_node = path[i+1]
            edges = sorted (G[this_node][next_node].values(), key=lambda x:x["w"])

            if len(edges)>1: #there are multiple edges between two nodes. we get edges with minumum weights
                minimum_weight = edges[0]["w"]
                edges = [edge for edge in edges if edge["w"]==minimum_weight]
            
            if this_path_all_combinations == []:
                for edge in edges:
                    info = [(this_node,next_node,edge)]
                    this_path_all_combinations.append (info)
            else:
                comb = []
                for edge in edges:
                    info = (this_node,next_node,edge)
                    temp = copy.deepcopy (this_path_all_combinations) #<<<CRITICAL>>> : deepcopy is really criticall here
                    for item in temp:
                        item.append (info)
                    comb.extend (temp)
                this_path_all_combinations = comb
        final_results.extend (this_path_all_combinations)
    return sorted (final_results, key=lambda x:len(x)) #<<<CRITICAL>>>: sort by number of edges in the path! 

if __name__ == "__main__":
    """
    G = nx.MultiGraph()
    G.add_nodes_from ([ (1, {"txt":"a"}) , 
                        (2, {"txt":"b"}) , 
                        (3, {"txt":"c"}) , 
                        (4, {"txt":"d"}) , 
                        (5, {"txt":"e"}) , 
                        (6, {"txt":"f"}) , 
                        (7, {"txt":"g"}) ,
                      ])
                        
    G.add_edge(1,2,w=3,tp="12")
    
    G.add_edge(2,3,w=1,tp="23_1")
    G.add_edge(2,3,w=2,tp="23_2")
    G.add_edge(3,6,w=2,tp="36")
    
    G.add_edge(3,4,w=1,tp="34")
    G.add_edge(4,5,w=1,tp="45")
    
    G.add_edge(5,6,w=1,tp="56_1")
    G.add_edge(5,6,w=2,tp="56_2")
    
    G.add_edge(1,5,w=8,tp="15")
    G.add_edge(1,6,w=16,tp="16")
    G.add_edge(6,6,w=16,tp="66")
    G.add_edge(1,7,w=1,tp="17")
    G.add_edge(7,6,w=1,tp="17")

    #paths1 = ORIGINAL_all_simple_paths_multigraph_CUTOFF (G,1,6,cutoff=None )
    #paths2 = sorted(_all_simple_paths_multigraph_CUTOFF_YIELDEDGES_AND_WEIGHTS (G,1,6,cutoff=None), key=lambda x:x[0])
    paths3 = _all_TopKPaths (G,1,6,None,None)
    #paths4 = _all_TopKPaths_ControlledMaxHeap (G, 1, 6, KTop=2,cutoff=None)
    paths5 = _all_TopKPaths_ControlledMaxHeap_Heuristic (G, 1, 6, KTop=5, cutoff=None)
    paths6 = _all_TopKPaths_ControlledMaxContainer_Heuristic (G, 1, 6, KTop=5, cutoff=None)
    for i in paths3:
        print i 
    print "-"*80

    for i in paths5:
        print i 
    print "-"*80
    
    for i in paths6:
        print i 
    print "-"*80
    """
    
    G = nx.MultiGraph()
    G.add_edge(0,1,w=1)
    G.add_edge(1,2,w=1,tp="12a")
    G.add_edge(1,2,w=2,tp="12b")
    G.add_edge(1,2,w=1,tp="12c")
    G.add_edge(2,3,w=1,tp="23")
    G.add_edge(3,4,w=1,tp="34a")
    G.add_edge(3,4,w=2,tp="34b")
    G.add_edge(4,5,w=1,tp="45a")
    G.add_edge(4,5,w=2,tp="45b")
    G.add_edge(4,5,w=1,tp="45c")
    G.add_edge(3,5,w=2,tp="35")
    G.add_edge(5,6,w=1)
    G.add_edge(0,6,w=6)
    
    """
    for i in all_shortest_paths_edges (G,0,6):
        print [sum(x[2]["w"] for x in i),i ]
    print "-"*80,"\n"
    """
    P_KTOP = 40
    P_CUTF = None
    X = _all_TopKPaths_ControlledMaxContainer_Heuristic (G, 0, 6, KTop=P_KTOP, cutoff=P_CUTF)
    Y = _all_TopKPaths_ControlledMaxContainer_Heuristic_GuarantySP (G, 0, 6, KTop=P_KTOP, cutoff=P_CUTF)

    for i in X:
        print i 
    print "-"*80
    for i in Y:
        print i 
    print "-"*80
