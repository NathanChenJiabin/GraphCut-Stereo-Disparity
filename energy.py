import maxflow


class Energy(object):

    def __init__(self, hintNbNodes=0, hintNbArcs=0):
        self.graph = maxflow.Graph[int](hintNbNodes, hintNbArcs)
        self.Econst = 0

    def add_term1(self, node_id, weight_s, weight_t):
        """
        Add a term E(x) of one binary variable to the energy function, where
        E(0)=E0, E(1)=E1. E0 and E1 can be arbitrary.
        :param node_id:
        :param weight_s:
        :param weight_t:
        :return: void
        """
        self.graph.add_tedge(node_id, weight_t, weight_s)

        return

    def add_variable(self, weight_s, weight_t):
        """
        Add a new binary variable
        :param weight_s:
        :param weight_t:
        :return: int, node id
        """
        node_id = self.graph.add_nodes(1)[0]
        # print("we add node: " +str(node_id))
        self.add_term1(node_id, weight_s, weight_t)

        return node_id

    def add_constant(self, cst):
        """
        Add a constant to the energy function
        :param cst: int, constant value to be added
        :return: void
        """
        self.Econst += cst
        return

    def add_term2(self, node_x, node_y, E00, E01, E10, E11):
        """
        Add a term E(x,y) of two binary variables to the energy function
        :param node_x:
        :param node_y:
        :param E00:
        :param E01:
        :param E10:
        :param E11:
        :return: void
        """
        self.graph.add_tedge(node_x, E11, E01)
        self.graph.add_tedge(node_y, 0, E00 - E01)
        self.graph.add_edge(node_x, node_y, 0, E01 + E10 - E00 - E11)
        return

    def forbid01(self, node_x, node_y):
        """
        Forbid (x,y)=(0,1) by putting infinite value to the arc from x to y
        :param node_x:
        :param node_y:
        :return: void
        """
        self.graph.add_edge(node_x, node_y, int(2**31 - 1), 0)
        return

    def minimize(self):
        """
        After construction of the energy function, call this to minimize it.
        :return: the minimum of the function
        """
        return self.Econst + self.graph.maxflow()

    def get_var(self, node_x):
        """
        After 'minimize' has been called, determine the value of variable 'node_x'
        in the optimal solution.
        The method get_segment returns 1 when the given node belongs to the partition of the source node
        (i.e., the minimum cut severs the terminal edge from the node to the sink),
        or 0 otherwise (i.e., the minimum cut severs the terminal edge from the source to the node).
        :param node_x: node id
        :return: 0 or 1
        """
        return int(self.graph.get_segment(node_x))
