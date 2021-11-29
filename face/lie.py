import os
os.system('python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/webface_r50')
os.system('mv work_dirs/webface_r50 work_dirs/webface_r50_lambda0.0003')
os.system('python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/webface_r50')