# we decide hyper-parameters based on page 7 of https://arxiv.org/pdf/1710.10903.pdf (transductive learning)

IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 ../train.py -m 'key=EnGAT_CiteSeer' \
     'EnGAT_CiteSeer.n_head=8' \
     'EnGAT_CiteSeer.n_head_last=1' \
     'EnGAT_CiteSeer.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'EnGAT_CiteSeer.dropout=choice(0.,0.4,0.6)' \
     'EnGAT_CiteSeer.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'EnGAT_CiteSeer.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'EnGAT_Cora.n_hid=choice(8, 16)' \
     'EnGAT_CiteSeer.pickup_ratio=choice(0.1, 0.2, 0.3, 0.4)' \
     'EnGAT_CiteSeer.strategy=choice(random, low_degree)' \
     'EnGAT_Cora.edge_method=choice(hybrid)' \
     'EnGAT_CiteSeer.graphs_number=choice(10, 20, 30, 40, 50, 60)' \
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done