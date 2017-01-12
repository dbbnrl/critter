def build():
    i = Input(shape=(224, 224, 1))
    m = i
    m = conv(m, 16, dim=5) # 5
    m = conv(m, 16, subsample=(2, 2)) # +2=7
    m = conv(m, 16) # +4=11
    m = conv(m, 16, subsample=(2, 2)) # +4=15
    m = conv(m, 32) # +8=23
    m = conv(m, 32, subsample=(2, 2)) # +8=31
    m = conv(m, 32) # +16=47
    m = conv(m, 32, subsample=(2, 2)) # +16=63
    m = conv(m, 64) # +32=95
    m = conv(m, 64, subsample=(2, 2)) # +32=127
    m = conv(m, 64) # +64=191
    m = conv(m, 64) # +64=255
#conv(model, 16, dim=1) # 255
    m = finalmax(m)
    o = m
    return i, o
