import networkx as nx
import matplotlib as mpl
from transport import *
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def normalize_unit(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def adjMatrixEdges(M, a_nodes, b_nodes=None):
    """
    Creates the edges in a NetworkX graph given some adjacency matrix
    :param M: adjacency matrix
    :param a_nodes: one group of nodes
    :param b_nodes: second group of nodes
    :return:
    """

    #Get the cordinates where transport is happening
    x, y = np.where(M > 0)

    assert len(x) == len(y), "these should be the same"

    #Easier on the eyes
    m = len(x)

    #Draw edges from A to A (same sample case)
    if b_nodes is None:
        edges = [(a_nodes[x[i]], a_nodes[y[i]], M[x[i]][y[i]]) for i in range(m)]

    #Draw edges from A to B
    else:
        edges = [(a_nodes[x[i]][0], b_nodes[y[i]][0], M[x[i]][y[i]]) for i in range(m)]

    return edges




class Map:
    def __init__(self, transport_map, X1, X2=None, symmetric=False, groups=None):
        """
        Class that contains the actual graph that I'm creating

        :param X1: the features for one sample of size n , np.ndarry n x (num_features)
        :param X2: the features for the otner sample of size m, np.ndarry m x (num_features)
        :param transport: n x m matrix, np.ndarry n x m
        :param symmetric: boolean denoting if the samples are the same items
        :param groups: denote if the nodes are to be grouped, or if this is an 1-to-1 graph
        """
        self.A = X1
        self.B = X2



        self.transport_map = transport_map
        self.symmetric = symmetric
        self.group_graph = None
        self.normalized = True

        if self.B is None:
            # Assume graph is symmetric if only one sample is recieved
            # self.B = X1
            symmetric = True
        else:
            symmetric = False

            #Make groups
        self.groups = []


        #Make a graph
        self.graph = nx.DiGraph()


        """
        if this group param is None, then we are making an individual 1-to-1 directed graph 
        Each node corresponds to an individual and each edge describes which element is mapped to which other 
        element.  The edge weights denote the mass of transport  
        """
        #Make graph
        A_nodes = [("A%d" % i, {"feature": self.A[i]}) for i in range(len(self.A))]

        if symmetric:

            #Quick check that the samples are the same size, if they're symetric
            # assert len(self.A) == len(self.B), "samples must be of the same length"

            #Add to graph
            # self.graph.add_nodes_from(A_nodes)
            for i in range(len(self.A)):
                self.graph.add_node("A%d" % i, feature=self.A[i])

            #Get edges
            edges = adjMatrixEdges(self.transport_map, list(self.graph.nodes))

            #Add Edges to graph
            self.graph.add_weighted_edges_from(edges)
        else:
            #Graph is not symetric so A, B are diff. sets of nodes
            B_nodes = [("B%d" % i, {"feature": self.B[i]}) for i in range(len(self.B))]

            self.graph.add_nodes_from(A_nodes, bipartite=0)
            self.graph.add_nodes_from(B_nodes, bipartite=1)

            # Get edges
            edges = adjMatrixEdges(self.transport_map, A_nodes, B_nodes)

            # Add Edges to graph
            self.graph.add_weighted_edges_from(edges)

    def add_group(self, rule, name=None):

        #Get nodes as tuple (name, data_dict)
        nodes = self.graph.nodes(data=True)

        #Get the names of the nodes that are grouped according to a rule
        #Get the n[0] (name) according to the filter over features n[1]
        grouped = [n[0] for n in nodes if rule(n[1]['feature'])]

        #Add to the big list of groups
        if grouped not in self.groups:
            self.groups.append((name, grouped))

        if len(self.groups) >= 2:
            self.create_grouped_graph()


    def create_grouped_graph(self):
        assert len(self.groups) >= 2, "must have at least two groups to create this graph"

        #For brevity
        M = self.transport_map
        groups = self.groups
        num_groups = len(groups)

        #Make the graph
        group_graph = nx.DiGraph()

        #Add a node for each group
        for i in range(num_groups):
            group_name, group_nodes = groups[i]
            if group_name is None:
                group_name = "G%d" % i
            group_graph.add_node(group_name, nodes=group_nodes)

        #Compute edge weights in adjacency matrix
        group_adjacency_matrix = np.zeros((num_groups, num_groups))
        for node in self.graph:
            side, index = node[0], int(node[1:])


            #Get the indices of a match
            if side == 'A':
                #Get all the groups that the node is in
                node_groups = [i for i in range(len(groups)) if node in groups[i][1]]

                #Get all the nodes that it is matched to
                match_indices = get_matches(index, mapping=M)

                #Determine the side of the matched node
                if self.symmetric:
                    match_side = "A"
                else:
                    match_side = "B"

                """
                Below we're making the adjacency matrix for the graph representation 
                Its kinda so it gets a block comment 
                First we're going to iterate through all the indices that our 'node' (defined above) is matched to 
                Then we get the groups for each of those matches - a match can be in multiple groups
                
                Finally, we update the adjacency matrix using the group indices, with the weight of the transport 
                from the individual level using the individual indices 
                """
                # all of the indices in the original graph 'node' is matched to,
                # match_indices[0] because its a tuple for some reason? the indices are in the 0 spot
                for m_i in match_indices[0]:

                    match_groups_indices = [i for i in range(len(groups)) if "%s%d" % (match_side, m_i) in groups[i][1]]
                    for m_g_i in match_groups_indices:
                        for n_g in node_groups:
                            group_adjacency_matrix[n_g][m_g_i] += M[index][m_i]

        self.group_adj = group_adjacency_matrix

        #Get the edges for the group graph
        group_edges = adjMatrixEdges(group_adjacency_matrix, list(group_graph.nodes))

        #Add the edges to the group graph
        for u, v, weight in group_edges:
            group_graph.add_edge(u, v, weight=weight)

        self.group_graph = group_graph

        if self.group_graph is not None and self.group_adj is not None and self.normalized:
            self.normalize_weights()

        return group_graph

    def resetGroupEdges(self):
        assert self.group_graph is not None and self.group_adj is not None, "Group Graph not instantiated"
        self.group_graph = nx.create_empty_copy(self.group_graph, with_data=True)
        group_edges = adjMatrixEdges(self.group_adj, list(self.group_graph.nodes))
        for u, v, weight in group_edges:
            self.group_graph.add_edge(u, v, weight=weight)


    def normalize_weights(self):
        if self.group_graph is not None:
            self.group_adj = normalize(self.group_adj, axis=1, norm="l1")

        self.resetGroupEdges()

        return self.group_adj


    def show_graph(self,
                   grouped=False,
                   sizeNodesBy=None,
                   numNodeSizes=3,
                   nodeSizeConstanst=50,
                   nodeSizeSmallest=1000,
                   nodeColorRule=None,
                   nodeSizeRule=None,
                   numNodeColors=2,
                   edgeColorRule=None,
                   ):

        if grouped:
            assert self.group_graph is not None, "Must have a group graph!"
            G = self.group_graph
        else:
            G = self.graph

        for u, v, d in G.edges(data=True):
            d['label'] = "%.3f" % d.get('weight', '')
            d['arrowsize'] = .6

        if grouped:
            for name, feature in G.nodes(data=True):
                lol = 2
                G.nodes[name]['height'] = lol
                G.nodes[name]['width'] = lol

        if grouped:
            G.graph['K'] = 3


        A = nx.nx_agraph.to_agraph(G)

        # A.layout(prog='dot')
        A.layout(prog="fdp")

        A.draw("simple.png")

        img = Image.open("simple.png")
        plt.figure(figsize=(12, 10))

        plt.imshow(img)
        plt.show()


        # pos = nx.layout.spring_layout(G)
        #
        # nodes = G.nodes(data=True)
        # if sizeNodesBy is not None:
        #     nodes = sizeNodesBy(nodes)
        #
        # node_colors_ = []
        # color_marker = 0
        #
        # labels = None
        #
        # node_sizes = []
        # size_marker = 0
        #
        # if self.symmetric:
        #
        #     for i in range(len(nodes)):
        #         if i >= len(nodes)*(size_marker + 1)/(numNodeSizes):
        #             size_marker += 1
        #         node_sizes.append(nodeSizeSmallest + ((size_marker + 1) * nodeSizeConstanst))
        #         if nodeColorRule is None:
        #             if i >= len(nodes)*(color_marker + 1)/(numNodeColors):
        #                 color_marker += 1
        #             node_colors_.append((color_marker + 1) / (numNodeColors))
        #         else:
        #             node_colors_.append(nodeColorRule(nodes[i]))
        #
        # #Not symmetric case
        # else:
        #
        #     if nodeColorRule is not None:
        #         node_colors_ = [nodeColorRule(nodes[i]) for i in range(len(nodes))]
        #     else:
        #         node_colors_ = [1] * len(nodes)
        #     if nodeSizeRule is None:
        #         node_sizes = [nodeSizeSmallest] * len(nodes)
        #     else:
        #         node_sizes = [nodeSizeRule(nodes[i]) for i in range(len(nodes))]
        #
        #
        #
        # M = G.number_of_edges()
        # edge_cmap = plt.get_cmap("Greys", lut=M)
        # edge_weights = np.array([(e[-1] + .5)/.6 for e in G.edges.data("weight")])
        # edge_colors = [edge_cmap(e_n) for e_n in edge_weights]
        #
        # node_cmap = plt.get_cmap(name='viridis', lut=len(G))
        # node_colors = [node_cmap(c) for c in node_colors_]
        #
        # nx.draw(
        #     G,
        #     pos=pos,
        #     with_labels=True,
        #     node_size = node_sizes,
        #     node_color = node_colors,
        #     alpha=.4
        # )
        #
        #
        # nx.draw_networkx_edges(
        #     G,
        #     pos,
        #     node_size=node_sizes,
        #     arrowstyle="->",
        #     arrowsize=10,
        #     edge_color=edge_colors,
        #     edge_cmap=plt.cm.Blues,
        #     width=2,
        # )
        #
        # labels = nx.get_edge_attributes(G, 'weight')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        #
        # ax = plt.gca()
        # ax.set_axis_off()
        # plt.show()

        #TODO: Draw group graph with labels











