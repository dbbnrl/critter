img_size = (224, 224)

conv(model, 16, dim=5, input_shape=(img_size+ (1,))) # 5
conv(model, 16, subsample=(2, 2)) # +2=7
conv(model, 16) # +4=11
conv(model, 16, subsample=(2, 2)) # +4=15
conv(model, 16) # +8=23
conv(model, 16, subsample=(2, 2)) # +8=31
conv(model, 16) # +16=47
conv(model, 16, subsample=(2, 2)) # +16=63
conv(model, 16) # +32=95
conv(model, 16, subsample=(2, 2)) # +32=127
conv(model, 16) # +64=191
conv(model, 16) # +64=255
#conv(model, 16, dim=1) # 255
finalmax(model)
