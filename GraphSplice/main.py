# This is an example to train a two-classes model.
import torch
import Biodata, GCNmodel
from multiprocessing import Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def foo(i):
        print(" This is Process ", i)


def main():
        for i in range(5):
                p = Process(target=foo, args=(i,))
                p.start()


if __name__ == '__main__':
    # ####################    Train   ########################
    # # main()
    # #
    # # data = Biodata.Biodata(fasta_file=".\Datasets\\Worm\\sequences_acceptor_400.fasta",
    # #                        label_file=".\Datasets\\Worm\\labels_acceptor.txt",
    # #                        feature_file = None
    # # #                        )
    # # data = Biodata.Biodata(fasta_file="donor_new.fasta",
    # #                        label_file="label_donor.txt",
    # #                        feature_file=None
    # #                        )
    # data = Biodata.Biodata(fasta_file="dataset/G3PO/acceptor_600.fasta",
    #                        label_file="G3PO_acc.txt",
    #                        feature_file= None
    #                        )
    # dataset = data.encode(thread=20)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GCNmodel.model(label_num=2, other_feature_dim=0).to(device)
    # GCNmodel.train(dataset, model, weighted_sampling=True)


    ##############   Test    ####################

    data2 = Biodata.Biodata(fasta_file="dataset/homo sapiens/sequences_acceptor_600.fasta",
                           label_file="dataset/homo sapiens/label_acceptor.txt",
                           feature_file=None
                           )

    data1 = data2.encode(thread=20)
    GCNmodel.test(data1)



