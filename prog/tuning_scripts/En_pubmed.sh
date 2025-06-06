# we decide hyper-parameters based on page 7 of https://arxiv.org/pdf/1710.10903.pdf (transductive learning)

IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 ../train.py -m 'key=EnGAT_PubMed' \
     'EnGAT_PubMed.n_head=8' \
     'EnGAT_PubMed.n_head_last=1' \
     'EnGAT_PubMed.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'EnGAT_PubMed.dropout=choice(0.,0.4,0.6)' \
     'EnGAT_PubMed.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'EnGAT_PubMed.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'EnGAT_PubMed.n_hid=choice(8, 16)' \
     'EnGAT_PubMed.pickup_ratio=choice(0.1, 0.2, 0.3, 0.4)' \
     'EnGAT_PubMed.strategy=choice(random, low_degree)' \
     'EnGAT_PubMed.edge_method=choice(shortest_path)' \
     'EnGAT_PubMed.graphs_number=choice(10, 20, 30, 40, 50, 60)' \
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done