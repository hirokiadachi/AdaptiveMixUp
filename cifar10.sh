read -p "Base network: " net
read -p "the number of epochs: " epochs
read -p "lr decay: " decay
for i in `seq 5`
do
    echo $i" trial"
    torchrun --nproc_per_node=8 --nnodes=1 --master_port=7000 main.py --epoch $epochs\
                    --num_trials $i\
                    --dataset cifar10\
                    --num_classes 10\
                    --arch $net\
                    --dataroot /home/workspace/Datasets/cifar10\
                    --lr 0.1\
                    --momentum 0.9\
                    --weight_decay 1e-4\
                    --checkpoint ./checkpoint\
                    --lr_decay $decay\
                    --train_batch_size 64\
                    --test_batch_size 256
done