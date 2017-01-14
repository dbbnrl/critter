from model import *

def dn_conv(model, filters, bias=False, weight_decay=1E-4, **kwargs):
    return conv(model, filters,
        activation=None, bias=bias,
        W_regularizer=l2(weight_decay), **kwargs)

def dn_convstep(model, filters, bottleneck=None, **kwargs):
    if bottleneck and (bottleneck < model.get_shape()[-1]):
        model = dn_conv(model, bottleneck, dim=1)
        model = bnorm(model)
        model = activation(model, 'relu')
    model = dn_conv(model, filters, **kwargs)
    model = bnorm(model)
    model = activation(model, 'relu')
    return model

def dn_dstep(model, in_filters, filters_per_layer, bottleneck=None):
    oldmodel = model
    model = dn_convstep(model, filters_per_layer, bottleneck=bottleneck)
    model = merge([oldmodel, model], mode='concat')
    return model, in_filters + filters_per_layer

def dn_dense(model, layers, in_filters, filters_per_layer, bottleneck=None):
    filters = in_filters
    for i in range(layers):
        model, filters = dn_dstep(model, filters, filters_per_layer, bottleneck=bottleneck)
    return model, filters

def dn_trans(model, filters=None):
    if filters:
        model = dn_conv(model, filters, dim=1)
    model = apool(model, dim=2)
    if filters:
        model = bnorm(model)
        model = activation(model, 'relu')
    return model

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
