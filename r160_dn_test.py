from model import *

def dn_convstep(model, filters, **kwargs):
    model = dn_conv(model, filters, **kwargs)
    model = bnorm(model)
    model = activation(model, 'relu')
    return model

def dn_dense(model, layers, in_filters, filters_per_layer):
    filters = in_filters
    for i in range(layers):
        oldmodel = model
        model = dn_convstep(model, filters_per_layer)
        model = merge([oldmodel, model], mode='concat')
        filters += filters_per_layer
    return model, filters

def dn_trans(model, filters):
    #model = dn_conv(model, filters, dim=1)
    model = apool(model, dim=2)
    #model = bnorm(model)
    #model = activation(model, 'relu')
    return model

def build():
    i = Input(shape=(160, 160, 1))
    m = i
    nf = 16
    g = 8
    m = dn_convstep(m, nf, dim=5, subsample=(2, 2))
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g)
    #nf //= 2 # compression
    m = dn_trans(m, nf)
    m, nf = dn_dense(m, 2, nf, g)
    m = dn_conv(m, 1, dim=1, bias=True)
    m = gmpool(m)
    #m = flatten(m)
    m = activation(m, 'sigmoid')
    o = m
    return i, o
