import heapq_max 
    #https://github.com/he-zhe/heapq_max
    #pip install --upgrade heapq_max

class CONTROLLED_MAX_HEAP:
    #receives values v like (actual_value,whatever_info)
    def __init__ (self, max_item_count):
        assert isinstance(max_item_count,int)
        assert max_item_count > 0 
        self.__max_item_count = max_item_count
        self.__heap = [] 
        
    def max(self):
        import numpy as np 
        if len(self.__heap) < self.__max_item_count:
            return np.inf
        else:                    
            return self.__heap[0][0]
    
    def pop(self):
        if self.__heap <> []:
            return heapq_max.heappop_max (self.__heap)
        else:
            return None
    
    def push(self,value):
        #print "****" ,value 
        if len(self.__heap) < self.__max_item_count:
            heapq_max.heappush_max (self.__heap, value)
        else:
            if self.__heap[0][0] <= value[0]: #a bigger value is trying to come inside, dont let it
                pass 
            else:
                heapq_max.heappop_max (self.__heap) #throw away
                heapq_max.heappush_max (self.__heap,value) #add new value
    
    def len(self):
        return len(self.__heap)
    
    def get_all(self):
        return sorted(self.__heap,key = lambda x:x[0])


class CONTROLLED_MAX_CONTRAINER:
    #receives values v like (actual_value,whatever_info)
    #AVOIDS ADDING SAME ITEM TWICE IN THE HEAP!!!
    def __init__ (self, max_item_count):
        assert isinstance(max_item_count,int)
        assert max_item_count > 0 
        self.__max_item_count = max_item_count
        self.__heap = [] 
        
    def max(self):
        import numpy as np 
        if len(self.__heap) < self.__max_item_count:
            return np.inf
        else:                    
            return self.__heap[-1][0]
    
    def push(self,value):
        #print "****" ,value 
        if len(self.__heap) < self.__max_item_count:
            if not value in self.__heap:
                self.__heap.append (value)
                self.__heap.sort (key= lambda x:x[0])
        else:
            if self.__heap[-1][0] < value[0]: #a bigger value is trying to come inside, dont let it
                pass 
            else:
                if not value in self.__heap:
                    self.__heap.append (value)
                    self.__heap.sort (key= lambda x:x[0])
                    temp = self.__heap[0:self.__max_item_count]
                    maxv = temp[-1][0]
                    temp+= [i for i in self.__heap[self.__max_item_count:] if i[0]==maxv]
                    self.__heap = temp
    
    def get_all(self):
        return self.__heap


if __name__ == "__main__":
    a = CONTROLLED_MAX_HEAP(3)
    a.push ((10,"101"))
    print a.get_all()
    a.push ((5 ,"5"))
    print a.get_all()
    a.push ((10,"102"))
    print a.get_all()
    a.push ((11 ,"11"))
    print a.get_all()
    a.push ((4 ,"41"))
    print a.get_all()
    a.push ((4 ,"42"))
    print a.get_all()
    a.push ((4 ,"43"))
    print a.get_all()
    a.push ((4 ,"44"))
    print a.get_all()
    a.push ((4 ,"45"))
    print a.get_all()
    a.push ((4 ,"46"))
    print a.get_all()
    a.push ((3 ,"3"))
    print a.get_all()
    a.push ((9 ,"9"))
    print a.get_all()
    a.push ((2 ,"2"))
    print a.get_all()
    a.push ((1 ,"1"))
    print a.get_all()
    
