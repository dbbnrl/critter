img_size = (160, 160)

conv(model, 16, dim=5, input_shape=(img_size+ (1,))) # 5
conv(model, 16, subsample=(2, 2)) # +2=7
conv(model, 32) # +4=11
conv(model, 32, subsample=(2, 2)) # +4=15
conv(model, 32) # +8=23
conv(model, 32, subsample=(2, 2)) # +8=31
conv(model, 32) # +16=47
conv(model, 32, subsample=(2, 2)) # +16=63
conv(model, 64) # +32=95
conv(model, 64, subsample=(2, 2)) # +32=127
#conv(model, 64) # +64=191
# conv(model, 64) # +64=253
#conv(model, 16, dim=1) # 253
#finalmax(model)
flatten(model)
dropout(model, 0.2)
dense(model, 16)
dropout(model, 0.5)
finaldense(model)
