from model import *
def build():
    i = Input(shape=(192, 192, 1))
    m = i
    g = 8
    nf = 2*g
    m = dn_conv(m, nf, dim=7, subsample=(2, 2))
    m = bnorm(m)
    m = activation(m, 'relu')
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, mode=apool)
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, mode=apool)
    m, nf = dn_dense(m, 4, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, mode=apool)
    m, nf = dn_dense(m, 4, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, mode=apool)
    m, nf = dn_dense(m, 6, nf, g)
    #m = dropout(m, 0.3)
    #m = gmpool(m)
    #m = dense(m, 1, activation='sigmoid', W_regularizer=l2(1E-4), b_regularizer=l2(1E-4))
    m = dn_conv(m, 1, dim=1, bias=True)
    m = gmpool(m)
    m = activation(m, 'sigmoid')
    o = m
    return i, o
