# Adaptive-Mixup
Adaptive MixUp (AdaMixUp) is proposed by H. Guo et al. on AAAI2019. [[arXiv](https://arxiv.org/abs/1809.02499)]\
AdaMixUp is the novel mixup method, which construct with policy region generator and intrusion discriminator.
Policy region generator provides three values that are alpha, delta, and discarded alpha by inputting two data for mixing.
Then, one defines policy region (alpha, delta+alpha), where delta+alpha <= 1 and discarded alpha is handled as offset.
