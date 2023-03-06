import math
from graphviz import Digraph
from graphviz import nohtml
import os
import string
import random

from ICPM2022.plotFig import gradient_color

os.environ["PATH"] += os.pathsep + r'D:\PyCharm 2019.2.2\Graphviz\bin'

class Node:
    def __init__(self, parent=None, children=None, data=None, tag=None, value=None):
        """
        node data structure
        :param parent
        :param children
        :param data: string
        """
        if children is None:
            children = []
        self.tag = tag if tag is not None else ''.join(random.sample(string.ascii_letters + string.digits, 8))
        self.parent = parent
        self.data = data
        self.children = children
        self.value = value

class Tree:
    def __init__(self, rootdata):
        self.root = Node(data=rootdata)

    def insert(self, parent_node, children_node):
        children_node.parent = parent_node
        parent_node.children.append(children_node)

    def search(self, node, data):
        """
        Take node as the root to find the node with the value of data, and return
        :param node: root
        :param data
        :return:
        """
        if node.data == data:
            return node
        elif len(node.children) == 0:
            return None
        else:
            for child in node.children:
                res = self.search(child, data)
                if res is not None:
                    return res
            return None

    def searchOne(self, node, data):
        if len(node.children) == 0:
            return None
        else:
            for i in node.children:
                if i.data == data:
                    return i
            return None

    def get_leavesByDataRoute(self, data_list):
        """
        Find all leaf collections under the path according to the list provided by the data field
        :param data_list: List of data fields for path，[Child node data， ……]
        :return: Leaf data collection under this path
        """
        leaves = set()
        node = self.root
        for data in data_list:
            node = self.search(node, data)

        for child in node.children:
            if len(child.children) > 0:
                leaves.add(child.data)

        return leaves

    def show(self, color_sum=20, minV=0, distance=0, save_path=None):
        """
        Show the tree structure
        :return:
        """
        from random import sample
        shapes = ['polygon','ellipse','triangle','circle','diamond','egg','trapezium','house','pentagon','hexagon','doublecircle','invtriangle','doubleoctagon','invtrapezium','tripleoctagon','octagon','invhouse','cds','star','plain']
        shapeCount = []
        shapeRecord = {}
        colors = gradient_color(['#009966','#FFFFFF'], color_sum)#"#4682B4", "#FFFAFA"
        colors1 = ['yellow2','antiquewhite1', 'aquamarine1', 'brown1', 'cadetblue1', 'chartreuse1', 'chocolate1', 'coral1', 'cornflowerblue','darkorchid1','sienna','royalblue','palevioletred','olivedrab1','mediumpurple1','lightslategray','lavenderblush2','indianred1','hotpink1','greenyellow','gold1','darkseagreen1']
        # plt = Digraph(comment='Tree')
        plt = Digraph(node_attr={'shape': 'record'})
        def print_node(node):
            # color = sample(colors, 1)[0]
            if len(node.children) > 0:
                for child in node.children:
                    cv = child.value.split(' : ')
                    i = math.ceil((float(cv[1])-minV)/distance)
                    if i == color_sum:
                        i -= 1
                    # print(color_sum-1-i)
                    if cv[0] not in shapeRecord.keys():
                        shapeCount.append(0)
                        shapeRecord.update({cv[0]: len(shapeCount)})
                    # child.value color=colors[color_sum-1-i]
                    plt.node(child.tag, "<FeaId> "+cv[0]+" |<MAE>"+cv[1], style='filled', color=colors1[shapeRecord[cv[0]]],
                    fillcolor=colors1[shapeRecord[cv[0]]] + ':' + colors[color_sum - 1 - i] + ';0.6')

                    # c + colors[color_sum - 1 - i]
                    #, shape=shapes[shapeRecord[cv[0]]]
                    plt.edge(node.tag, child.tag)
                    print_node(child)

        cv = self.root.value.split(' : ')
        i = math.ceil((float(cv[1]) - minV) / distance)
        if i == color_sum:
            i -= 1
        plt.node(self.root.tag, "<FeaId>"+cv[0]+"|<MAE>"+cv[1], style='filled', color=colors[color_sum-1-i])
        print_node(self.root)
        plt.view()

        if save_path is not None:
            plt.render(save_path)


def create(csvfile_name):
    tree = Tree('0')
    with open(csvfile_name, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
        for line in lines:
            elements = line.split(',')
            p = tree.root
            for i in range(len(elements) - 1):
                if len(elements[i]) == 0:
                    continue
                q = tree.search(p, elements[i])
                if q is None:
                    q = Node(data=elements[i])
                    tree.insert(p, q)
                p = q
            for i in range(len(elements) - 1, len(elements)):
                if len(elements[i].strip()) == 0:
                    continue
                new_node = Node(data=elements[i])
                tree.insert(p, new_node)

    return tree


def tree_test():
    tree = Tree('10')
    root = tree.root
    for i in range(7, 10):
        node = Node(data=str(i))
        tree.insert(root, node)

    p = tree.search(root, '7')
    child = Node(data=u'淡红')
    tree.insert(p, child)

    p = tree.search(root, '8')
    for i in range(4, 6):
        node = Node(data=str(i))
        tree.insert(p, node)

    tree.show()


if __name__ == '__main__':
    file = r'0.csv'
    tree = create(file)
    tree.show()

    tree = Tree('0')
    tree.insert('1','2')
    tree.show()
