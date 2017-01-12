def build():
    i = Input(shape=(224, 224, 1))
    m = i
    m = conv(m, 16, input_shape=(img_size+ (1,)))
    m = pool(m)
    m = conv(m, 16)
    m = pool(m)
    m = conv(m, 16)
    m = pool(m)
    m = conv(m, 16)
    m = pool(m)
    m = flatten(m)
    m = dense(m, 16)
    m = finaldense(m)
    o = m
    return i, o
