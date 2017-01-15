from model import *
def build():
    i = Input(shape=(224, 224, 1))
    m = i
    nf = 16
    g = 8
    m = dn_convstep(m, nf, dim=7, subsample=(2, 2))
    m, nf = dn_dense(m, 1, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m)
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m)
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m)
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m)
    m, nf = dn_dense(m, 2, nf, g)
    m = gapool(m)
    m = dense(m, 1, activation='sigmoid', W_regularizer=l2(1E-4), b_regularizer=l2(1E-4))
    #m = dn_conv(m, 1, dim=1, bias=True)
    #m = gmpool(m)
    #m = activation(m, 'sigmoid')
    o = m
    return i, o
