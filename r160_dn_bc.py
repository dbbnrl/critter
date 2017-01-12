from model import *
def build():
    i = Input(shape=(160, 160, 1))
    m = i
    nf = 16
    g = 8
    m = conv(m, nf, dim=5, subsample=(2, 2))
    m = bnorm(m)
    m, nf = dn_dense(m, 2, nf, g, bottleneck=16)
    nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g, bottleneck=16)
    nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g, bottleneck=16)
    nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g, bottleneck=16)
    nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g, bottleneck=16)
    m = activation(m, 'relu')
    m = gapool(m)
    m = finaldense(m)
    o = m
    return i, o
