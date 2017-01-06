img_size = (224, 224)

conv(model, 16, input_shape=(img_size+ (1,))) # 3x3
conv(model, 16, subsample=(2, 2)) # +2=5x5,/2
conv(model, 16) # +4=9x9
conv(model, 32, subsample=(2, 2)) # +4=13x13,/2
conv(model, 32) # +8=21x21
conv(model, 32, subsample=(2, 2)) # +8=29x29,/2
conv(model, 32) # +16=45x45
conv(model, 64, subsample=(2, 2)) # +16=61x61, /2
conv(model, 64) # +32=93x93
conv(model, 64, subsample=(2, 2)) # +32=125x125, /2
conv(model, 64) # +64=189x189
conv(model, 64, subsample=(2, 2)) # +64=253x253, /2
conv(model, 64) # +128=381x381
conv(model, 64, dim=1) # 381x381
finalmax(model)
