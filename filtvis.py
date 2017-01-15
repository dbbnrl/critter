from scipy.misc import imsave, imshow
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from matplotlib import pyplot
from math import sqrt, ceil

def make_imgrid(images):
    nimg = len(images)
    gdim = ceil(sqrt(nimg))
    img_dim = np.shape(images[0])[0]
    if len(np.shape(images[0])) < 3:
        gray = True
    else:
        gray = False

    # build a black picture with enough space for
    # our 4 x 4 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = gdim * img_dim + (gdim - 1) * margin
    height = gdim * img_dim + (gdim - 1) * margin
    if gray:
        grid_image = np.zeros((width, height), dtype='uint8')
    else:
        grid_image = np.zeros((width, height, 3), dtype='uint8')

    # fill the picture with our saved filters
    try:
        for i in range(gdim):
            for j in range(gdim):
                img = images[i * gdim + j]
                if gray:
                    grid_image[(img_dim + margin) * i: (img_dim + margin) * i + img_dim,
                               (img_dim + margin) * j: (img_dim + margin) * j + img_dim] = img
                else:
                    grid_image[(img_dim + margin) * i: (img_dim + margin) * i + img_dim,
                               (img_dim + margin) * j: (img_dim + margin) * j + img_dim, :] = img
    except IndexError:
        pass
    return grid_image

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    # x -= x.mean()
    # x /= (x.std() + 1e-5)
    # x *= 0.1
    # # clip to [0, 1]
    # x += 0.5

    mi = np.min(x)
    ma = np.max(x)
    rg = ma-mi
    x -= mi
    x /= rg

    return x

def to_bytes(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# # build the VGG16 network with ImageNet weights
# model = vgg16.VGG16(weights='imagenet', include_top=False)
# print('Model loaded.')

# model.summary()

def vis_optimage(model, layer_name):
    # this is the placeholder for the input images
    input_img = model.input
    img_size = model.input_shape[1:3]
    img_shape = (1,) + img_size + (1,)
    img_dim = img_size[0]

    layer = model.get_layer(layer_name)
    layer_input = layer.input
    kept_filters = []
    for filter_index in range(layer.input_shape[3]):
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_input[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        # grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        input_img_data = np.random.random(img_shape)
        input_img_data = (input_img_data - 0.5) * 2.65

        prev_loss = -100000000.
        last_progress = 0
        best_loss = prev_loss

        # we run gradient ascent for 20 steps
        while True:
            loss_value, grads_value = iterate([input_img_data, 0])
            progress = loss_value - prev_loss
            if progress > 0.:
                step *= 1.05
            else:
                step /= 1.2
            # if (loss_value - best_loss) > abs(best_loss / 10000.):
            if loss_value > best_loss:
                best_loss = loss_value
                last_progress = 0
            else:
                last_progress += 1
                if last_progress > 20:
                    break
            prev_loss = loss_value
            input_img_data += grads_value * step
            input_img_data = np.clip(input_img_data, -2.65, 2.65)

            print('Current loss value:', loss_value)
            #if loss_value <= 0.:
                ## some filters get stuck to 0, we can skip them
                #break

        # decode the resulting input image
        # if loss_value > 0:
        img = to_bytes(deprocess_image(input_img_data[0, :, :, 0]))
        kept_filters.append(img)
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # # we will stich the filters on a 4 x 4 grid.
    # n = 4

    # # build a black picture with enough space for
    # # our 4 x 4 filters of size 128 x 128, with a 5px margin in between
    # margin = 5
    # width = n * img_dim + (n - 1) * margin
    # height = n * img_dim + (n - 1) * margin
    # stitched_filters = np.zeros((width, height))

    # # fill the picture with our saved filters
    # try:
    #     for i in range(n):
    #         for j in range(n):
    #             img, loss = kept_filters[i * n + j]
    #             stitched_filters[(img_dim + margin) * i: (img_dim + margin) * i + img_dim,
    #                             (img_dim + margin) * j: (img_dim + margin) * j + img_dim] = img
    # except IndexError:
    #     pass

    grid_image = make_imgrid(kept_filters)
    # save the result to disk
    imsave('filters_%s.png' % layer_name, grid_image)
    # print(np.shape(stitched_filters))
    # pyplot.ion()
    # pyplot.imshow(grid_image)
    # pyplot.gray()
    # pyplot.show()
    #pyplot.waitforbuttonpress()

def vis_activations(model, layer_name, gen):
    # this is the placeholder for the input images
    input_img = model.input
    img_size = model.input_shape[1:3]
    img_shape = (1,) + img_size + (1,)
    img_dim = img_size[0]

    layer = model.get_layer(layer_name)
    layer_output = layer.output
    nfilters = layer.output_shape[3]

    afunc = K.function([input_img, K.learning_phase()], [layer_output])

    pyplot.ion()
    #pyplot.gray()

    while True:
        (in_img, *rest) = next(gen)
        in_img = np.reshape(in_img, img_shape)
        images = []
        [acts] = afunc([in_img, 0])
        while np.shape(acts)[1] < img_dim:
            acts = np.repeat(np.repeat(acts, 2, 1), 2, 2)
        i = deprocess_image(np.squeeze(in_img))
        i *= 0.2
        for f in range(nfilters):
            r = acts[0, :, :, f]
            r = deprocess_image(r)
            r += i
            aimg = np.stack([r, i, i], -1)
            aimg = to_bytes(aimg)
            images.append(aimg)
        grid_image = make_imgrid(images)
        #pyplot.imshow(grid_image, cmap=None, norm=None)
        # pyplot.gray()
        pyplot.imshow(grid_image)
        pyplot.show()
        pyplot.waitforbuttonpress()
