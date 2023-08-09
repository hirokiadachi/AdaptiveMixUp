import numpy as np
import argparse

def configures():
    parser = argparse.ArgumentParser()
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    
    ## training setting
    parser.add_argument("--offline", action="store_true")
    parser.add_argument('--epoch', type=int, default=100,
                        help='the number of epochs')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint',
                        help='directry name to save checkpoint')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='the number of batchs at the training time')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='the number of batchs at the test time')
    parser.add_argument('--arch', type=str, default='preactres18')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--seed', type=int, default=np.random.randint(4294967295),
                        help='random seed')

    ## Optimizer setting
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='amount of the momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--pretrained', type=str, default='',
                        help='pretrained model path for evalation.')
    parser.add_argument('--lr_decay', nargs="*", type=int, default=[400, 800])
    args = parser.parse_args()
    return args