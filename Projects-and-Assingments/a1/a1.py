
# coding: utf-8

# In[2]:
#The Program takes 23 seconds for execution, the slowness due to Recursion used for the bottom up approach

from collections import Counter, defaultdict, deque
from operator import itemgetter, attrgetter
from itertools import islice

from itertools import chain

import networkx as nx
import copy
import math
import networkx as nx
import urllib.request
import matplotlib.pyplot as plt

def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.
    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque
    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.
    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    In the doctests below, we first try with max_depth=5, then max_depth=2.
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    
    cntr=0         
    levellist=deque([root])
    Visits={root:cntr}    
    parent={root:[]} 
    while levellist:
        cntr+=1
        curlevel=levellist
        levellist=[]
        for v in curlevel:
            for w in graph.neighbors(v):
                if w not in Visits:
                    parent[w]=[v]
                    Visits[w]=cntr
                    levellist.append(w)
                elif (Visits[w]==cntr):
                    parent[w].append(v)
        if (max_depth and max_depth <= cntr):
            break
    parent_E=parent.copy()
    del parent[root]
    parent_E[root]=list(root)
    ss_nodes={}
    for k, v in parent_E.items():
        ss_nodes[k]=len(v)
    return Visits,ss_nodes,parent

def complexity_of_bfs(V, E, K):
    """
    If V is the number of vertices in a graph, E is the number of
    edges, and K is the max_depth of our approximate breadth-first
    search algorithm, then what is the *worst-case* run-time of
    this algorithm? As usual in complexity analysis, you can ignore
    any constant factors. E.g., if you think the answer is 2V * E + 3log(K),
    you would return V * E + math.log(K)
    >>> v = complexity_of_bfs(13, 23, 7)
    >>> type(v) == int or type(v) == float
    True
    """
    return(V + math.log(2*K))
    
def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...
    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge  (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).
      Any edges excluded from the results in bfs should also be exluded here.
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """
    parent_list=defaultdict(list)
    a=0
    vv=[]
    scores={}
    nodes=sorted(list(node2parents.keys()))
    values=sorted(list(node2parents.values()))
    values=(set(sum(values, [])))
    for x in nodes:
        if x not in values:
            vv.append(x)

    for k,v in node2parents.items():
        for x in v:
            parent_list[x].append(k)
    aa=0
    def val_of_edge(a):
        global aa 
        aa=0
        if(a in vv):
            return 1/node2num_paths[a]
        else:
            for i in parent_list[a]:
                aa+=val_of_edge(i)
            aa+=1
            return aa
    for k,v in node2parents.items():
        for y in v:
            s=tuple(sorted((k,y)))
            scores[s]=(val_of_edge(k))
    return scores

def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.
    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).
    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
    """
    ###TODO
    n=[]
    for i in graph.edges():
        n.append(tuple(sorted(i)))
    scores=dict.fromkeys(n, 0.0)
    for v in graph.nodes():
        node2distances, node2num_paths, node2parents=bfs(graph, v,max_depth)
        b2u=bottom_up(v, node2distances, node2num_paths, node2parents)
        for x,y in b2u.items():
            scores[x]+=y
    for key,value in scores.items():
        scores[key]=value/2
    return scores

def is_approximation_always_right():
    """
    Look at the doctests for approximate betweenness. In this example, the
    edge with the highest betweenness was ('B', 'D') for both cases (when
    max_depth=5 and max_depth=2).
    Consider an arbitrary graph G. For all max_depth > 1, will it always be
    the case that the edge with the highest betweenness will be the same
    using either approximate_betweenness verses the exact computation?
    Answer this question below.
    In this function, you just need to return either the string 'yes' or 'no'
    (no need to do any actual computations here).
    >>> s = is_approximation_always_right()
    >>> type(s)
    <class 'str'>
    """
    return 'no'

def partition_girvan_newman(graph, max_depth):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.
    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).
    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    Returns:
      A list of networkx Graph objects, one per partition.
      
    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """
    bet=approximate_betweenness(graph,max_depth)
    bet=sorted(bet.items(),key=lambda x:(-x[1],x[0]))
    #print("BB:",bet[:30])
    components = [c for c in nx.connected_component_subgraphs(graph)]
    for k in bet:
        #print("removing::",k)
        graph.remove_edge(k[0][0],k[0][1])
        components = [c for c in nx.connected_component_subgraphs(graph)]
        if(len(components) != 1):
            break
    #print("KKK:",k,len(components))
    return components

def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.
    Params:
    graph........a networkx graph
    min_degree...degree threshold
    Returns:
    a networkx graph, filtered as defined above.
    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    G=nx.Graph()
    for node in graph.nodes():
        if(graph.degree(node)>=min_degree):
            G.add_node(node)
            for n in graph.neighbors(node):
                if(graph.degree(n)>=min_degree):
                    G.add_edge(node,n)
    return G
 
def volume(nodes, graph):
    """
    Compute the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes.
    Params:
    nodes...a list of strings for the nodes to compute the volume of.
    graph...a networkx graph
    >>> volume(['A', 'B', 'C'], example_graph())
    4
    """
    result = []
    for each in nodes:
        for i in graph.neighbors(each):
            result.append((each,i))
            
    for each in result:
        if tuple(reversed(each)) in result:
            result.remove(tuple(reversed(each)))   
    return len(result)

def cut(S, T, graph):
    """
    Compute the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
    S.......set of nodes in first subset
    T.......set of nodes in second subset
    graph...networkx graph
    Returns:
    An int representing the cut-set.
    >>> cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    1
    """
    result = []
    for node in S:
        for n in T:
            if node in graph.neighbors(n):
                result.append(node)
    return len(result)


def norm_cut(S, T, graph):
    """
    The normalized cut value for the cut S/T. (See lec06.)
    Params:
    S.......set of nodes in first subset
    T.......set of nodes in second subset
    graph...networkx graph
    Returns:
    An float representing the normalized cut value
    """
    CUT = cut(S,T,graph)
    VOL_S = volume(S,graph)
    VOL_T = volume(T,graph)
    result = (CUT / VOL_S) + (CUT / VOL_T)
    return float(result)


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method
    we've developed, we will run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Recall that smaller norm_cut scores correspond to better partitions.
    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman
    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman. See Log.txt for an example.
    """
    ###TODO
    pass
    result = {}
    for i in max_depths:
        clusters = partition_girvan_newman(graph.copy(), i)
        #print("---")
        #print("Cluster 2 nodes of partition whose depth is ",i,":",clusters[1].nodes())
        result[i]=norm_cut(clusters[0].nodes(), clusters[1].nodes(), graph.copy())
    return(sorted(result.items()))


def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    As in lecture, we'll assume there is a test_node for which we will
    remove some edges. Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    E.g., if 'A' has neighbors 'B' and 'C', and n=1, then the edge
    ('A', 'B') will be removed.
    Be sure to *copy* the input graph prior to removing edges.
    Params:
    graph.......a networkx Graph
    test_node...a string representing one node in the graph whose
    edges will be removed.
    n...........the number of edges to remove.
    Returns:
    A *new* networkx Graph with n edges removed.
    In this doctest, we remove edges for two friends of D:
    >>> g = example_graph()
    >>> sorted(g.neighbors('D'))
    ['B', 'E', 'F', 'G']
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> sorted(train_graph.neighbors('D'))
    ['F', 'G']
    """
    train_graph = graph.copy()
    edges_to_rem = sorted(train_graph.edges(test_node))[:n]
    train_graph.remove_edges_from(edges_to_rem)
    return train_graph

def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.
    Note that we don't return scores for edges that already appear in the graph.
    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.
    In this example below, we remove edges (D, B) and (D, E) from the
    example graph. The top two edges to add according to Jaccard are
    (D, E), with score 0.5, and (D, A), with score 0. (Note that all the
    other remaining edges have score 0, but 'A' is first alphabetically.)
    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> jaccard(train_graph, 'D', 2)
    [(('D', 'E'), 0.5), (('D', 'A'), 0.0)]
    """
    ###TODO
    scores = []
    set_a = set(graph.neighbors(node))
    for n in graph.nodes():
        if n != node and not graph.has_edge(node, n):
            set_b = set(graph.neighbors(n))
            score = 1. * (len(set_a & set_b)) / (len(set_a | set_b))
            scores.append(((node, n), score))
        
    return sorted(scores,key=lambda x:(-x[1],x[0]))[:k]

def path_score(graph, node, k, beta):
    """
    Compute a new link prediction scoring function based on the shortest
    paths between two nodes, as defined above.
    Note that we don't return scores for edges that already appear in the graph.
    This algorithm should have the same time complexity as bfs above.
    Params:
      graph....a networkx graph
      root.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
      beta.....the beta parameter in the equation above.
    Returns:
      A list of tuples in descending order of score. Ties are broken by
      alphabetical order of the terminal node in the edge.
    In this example below, we remove edge (D, F) from the
    example graph. The top two edges to add according to path_score are
    (D, F), with score 0.5, and (D, A), with score .25. (Note that (D, C)
    is tied with a score of .25, but (D, A) is first alphabetically.)
    >>> g = example_graph()
    >>> train_graph = g.copy()
    >>> train_graph.remove_edge(*('D', 'F'))
    >>> path_score(train_graph, 'D', k=4, beta=.5)
    [(('D', 'F'), 0.5), (('D', 'A'), 0.25), (('D', 'C'), 0.25)]
    """
    scores = []
    node2distances, node2num_paths, node2parents=bfs(graph, node,k)
    for l in graph.nodes():
        if l != node and  not graph.has_edge(node, l):
            #print(l[:k])
            maxLen = node2distances[l]
            num_paths=node2num_paths[l]
            score=(beta**maxLen)*node2num_paths[l]
            scores.append(((node, l), score))
    return(sorted(scores,key=lambda x:(-x[1],x[0]))[:k])

def evaluate(predicted_edges, graph):
    """
    Return the fraction of the predicted edges that exist in the graph.
    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph
    Returns:
      The fraction of edges in predicted_edges that exist in the graph.
    In this doctest, the edge ('D', 'E') appears in the example_graph,
    but ('D', 'A') does not, so 1/2 = 0.5
    >>> evaluate([('D', 'E'), ('D', 'A')], example_graph())
    0.5
    """
    l=len(predicted_edges)
    evl=[]
    for x in predicted_edges:
        if graph.has_edge(*x):
            evl.append(x)
    return float(len(evl)/l)

def download_data():
    """
    Download the data. Done for you.
    """
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph. 
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')

def main():
    #The Program takes 23 seconds for execution, the slowness due to Recursion used for the bottom up approach

    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' % (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' % (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph.copy(), range(1,5)))
    clusters = partition_girvan_newman(subgraph.copy(), 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' % (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph.copy(), test_node, 5)
    print('train_graph has %d nodes and %d edges' % (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' % evaluate([x[0] for x in jaccard_scores], subgraph.copy()))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' % evaluate([x[0] for x in path_scores], subgraph.copy()))

if __name__ == '__main__':
    main()




