img_size = (224, 224)

conv(model, 16, input_shape=(img_size+ (1,)))
pool(model)
conv(model, 16)
pool(model)
conv(model, 16)
pool(model)
conv(model, 16)
pool(model)
flatten(model)
dense(model, 16)
finaldense(model)