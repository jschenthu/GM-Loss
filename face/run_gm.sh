CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/vggface_r50_gm
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_ijbc.py --model-prefix work_dirs/vggface_l0.00003_a0.01/backbone_70000.pth --image-path ijb/IJBB --batch-size 2048 --job vggface --target IJBB --gm
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_ijbc.py --model-prefix work_dirs/vggface_l0.00003_a0.01/backbone_70000.pth --image-path ijb/IJBC --batch-size 2048 --job vggface --target IJBC --gm

