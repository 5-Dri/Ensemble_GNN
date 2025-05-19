# we decide hyper-parameters based on page 7 of https://arxiv.org/pdf/1710.10903.pdf (transductive learning)

IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 ../train.py -m 'key=EnGAT_Cora' \
     'EnGAT_Cora.n_head=8' \
     'EnGAT_Cora.n_head_last=1' \
     'EnGAT_Cora.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'EnGAT_Cora.dropout=choice(0.,0.4,0.6)' \
     'EnGAT_Cora.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'EnGAT_Cora.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'EnGAT_Cora.pickup_tario=(0.1, 0.2, 0.3, 0.4)' \
     'EnGAT_Cora.strategy=choice(random, low_degree)' \
     'EnGAT_Cora.graphs_number=choice(10, 20, 30, 40)' \
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done