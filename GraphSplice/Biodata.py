import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
from functools import partial

import torch
import torch_geometric.transforms as T
import torch_geometric.utils as ut
from torch_geometric.data import Data
import encode_seq



class BipartiteData(Data):   # Bipartite graph data
    # 这个类继承了 PyTorch Geometric 的 Data 类，用于存储双分图数据
    def _add_other_feature(self, other_feature) :  # 这个函数用于添加其他特征
        self.other_feature = other_feature

    def __inc__(self, key, value):  # 这个函数用于返回图中节点的数量
        if key == 'edge_index':  # key是边的索引，value是边的数量
            return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
        # self.x_src.size(0) 是源节点特征矩阵中的节点数量
        # self.x_dst.size(0) 是目标节点特征矩阵中的节点数量
        # torch.tensor 是 PyTorch 中的张量
        else:
            return super(BipartiteData, self).__inc__(key, value)
            # 这里是调用父类的 __inc__ 函数


class GraphDataset():
    # 这个类用于处理图数据
    def __init__(self, pnode_feature, fnode_feature, other_feature, edge, graph_label):
        # 此方法用于初始化图数据
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.other_feature = other_feature
        self.edge = edge
        self.graph_label = graph_label
    def process(self):
        # 此方法用于处理图数据
        data_list = []  # 用于存储图数据
        for i in range(self.pnode_feature.shape[0]):  # self.pnode_feature.shape[0]为总DNA数量
            edge_index = torch.tensor(self.edge, dtype=torch.long)  # edge_index should be long type
            # 该张量存储了图的边的索引信息。这里要求边的索引数据类型为 long 类型。
            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)
            # 创建两个 PyTorch 张量 x_p 和 x_f，它们分别表示双分图中的两个节点集合的特征。这些特征来自原始数据的节点特征。
            if type(self.graph_label) == np.ndarray:  # np.ndarray 表示多维 Numpy 数组，表示存在图的标签
            # 根据输入的图标签数据（self.graph_label）的类型，进行不同的操作
                y = torch.tensor([self.graph_label[i]], dtype=torch.long)
                # 创建 PyTorch 张量 y，它表示图的标签。这里要求图的标签数据类型为 long 类型。
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)
                # 创建 BipartiteData 类的对象 data，用于存储图数据, x_src 表示源节点特征，x_dst 表示目标节点特征,
            else:  # 如果不存在图的标签
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, num_nodes=None)

            if type(self.other_feature) == np.ndarray:
                other_feature = torch.tensor(self.other_feature[i, :], dtype=torch.float)
                data._add_other_feature(other_feature)

            data_list.append(data)  # 将图数据添加到 data_list 中

        return data_list

class Biodata:
    def __init__(self, fasta_file, label_file, feature_file=None, K=3, d=1):
        self.dna_seq = {}  # 用于存储 DNA 序列
        for seq_record in SeqIO.parse(fasta_file, "fasta"):  # 读取 fasta 文件
            self.dna_seq[seq_record.id] = str(seq_record.seq)  # 将 DNA 序列添加到 self.dna_seq 中

        if feature_file is None:
            self.other_feature = None
        else:
            self.other_feature = np.loadtxt(feature_file)

        self.K = K
        self.d = d

        self.edge = []
        for i in range(4**(K*2)):
            a = i // 4**K
            b = i % 4**K
            self.edge.append([a, i])  # 边的起点节点索引
            self.edge.append([b, i])  # 边的终点节点索引
        # 这个for循环用于生成边的索引信息，这里的边是指双分图中的边
        self.edge = np.array(self.edge).T
        # 这行代码用于将边的索引信息存储为一个 2x(4^(K*2)) 的矩阵
        # T是转置，这里是将边的索引信息转置，这样就可以将边的索引信息存储为一个 2x(4^(K*2)) 的矩阵

        if label_file:
            self.label = np.loadtxt(label_file)  # 读取图的标签
        else:
            self.label = None

    def encode(self, thread):
        print("Encoding sequences...")
        seq_list = list(self.dna_seq.values())
        pool = Pool(thread)
        partial_encode_seq = partial(encode_seq.matrix_encoding, K=self.K, d=self.d)
        feature = np.array(pool.map(partial_encode_seq, seq_list))
        pool.close()
        pool.join()
        self.pnode_feature = feature.reshape(-1, self.d, 4 ** (self.K * 2))
        self.pnode_feature = np.moveaxis(self.pnode_feature, 1, 2)
        zero_layer = feature.reshape(-1, self.d, 4 ** self.K, 4 ** self.K)[:, 0, :, :]
        self.fnode_feature = np.sum(zero_layer, axis=2).reshape(-1, 4 ** self.K, 1)
        del zero_layer
        graph = GraphDataset(self.pnode_feature, self.fnode_feature, self.other_feature, self.edge, self.label)
        dataset = graph.process()
        # wwwww
        return dataset





