# we decide hyper-parameters based on page 7 of https://arxiv.org/pdf/1710.10903.pdf (transductive learning)

IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 ../train.py -m 'key=GAT_CiteSeer' \
     'GAT_CiteSeer.n_head=8' \
     'GAT_CiteSeer.n_head_last=1' \
     'GAT_CiteSeer.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_CiteSeer.dropout=choice(0.,0.4,0.6)' \
     'GAT_CiteSeer.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'GAT_CiteSeer.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_CiteSeer.n_hid=choice(8, 16)' \
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done