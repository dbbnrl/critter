def build():
    i = Input(shape=(224, 224, 1))
    m = i
    m = conv(m, 32, dim=5, input_shape=(img_size+ (1,))) # 5
    m = pool(m)
#conv(model, 32, subsample=(2, 2)) # +2=7
    m = conv(m, 32) # +4=11
    m = pool(m)
#conv(model, 32, subsample=(2, 2)) # +4=15
    m = conv(m, 32) # +8=23
    m = pool(m)
#conv(model, 32, subsample=(2, 2)) # +8=31
    m = conv(m, 32) # +16=47
    m = pool(m)
#conv(model, 32, subsample=(2, 2)) # +16=63
    m = conv(m, 64) # +32=95
    m = pool(m)
#conv(model, 64, subsample=(2, 2)) # +32=127
    m = conv(m, 64) # +64=191
#conv(model, 64) # +64=255
#conv(model, 16, dim=1) # 255
#finalmax(model)
    m = flatten(m)
    m = dropout(m, 0.2)
    m = dense(m, 64)
    m = dropout(m, 0.5)
    m = finaldense(m)
    o = m
    return i, o
