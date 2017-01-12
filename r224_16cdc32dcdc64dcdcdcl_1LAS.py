def build():
    i = Input(shape=(224, 224, 1))
    m = i
    m = conv(m, 16, input_shape=(img_size+ (1,))) # 3x3
    m = conv(m, 16, subsample=(2, 2)) # +2=5x5,/2
    m = conv(m, 16) # +4=9x9
    m = conv(m, 32, subsample=(2, 2)) # +4=13x13,/2
    m = conv(m, 32) # +8=21x21
    m = conv(m, 32, subsample=(2, 2)) # +8=29x29,/2
    m = conv(m, 32) # +16=45x45
    m = conv(m, 64, subsample=(2, 2)) # +16=61x61, /2
    m = conv(m, 64) # +32=93x93
    m = conv(m, 64, subsample=(2, 2)) # +32=125x125, /2
    m = conv(m, 64) # +64=189x189
    m = conv(m, 64, subsample=(2, 2)) # +64=253x253, /2
    m = conv(m, 64) # +128=381x381
    m = conv(m, 64, dim=1) # 381x381
    m = finalavg(m)
    o = m
    return i, o
