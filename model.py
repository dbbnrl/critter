from importlib import import_module
from keras.models import Sequential, Model, load_model, save_model, model_from_config
from keras.optimizers import SGD, Adam
from keras.layers import Activation, Input, merge, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import _to_tensor, _EPSILON
from keras import backend as K
import tensorflow as tf
from tensorflow.python.training.training import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
# from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
# from tensorflow.tools.quantize_graph import GraphRewriter

pos_weight = 5.0

def weighted_binary_crossentropy(output, target, pos_weight, from_logits=False):
    '''Binary crossentropy between an output tensor and a target tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.weighted_cross_entropy_with_logits(output, target, pos_weight)

def my_loss_fn(y_true, y_pred):
    return K.mean(weighted_binary_crossentropy(y_pred, y_true, pos_weight), axis=-1)

def convxy(model, filters, xdim, ydim,
           activation='relu', border_mode='same', **kwargs):
    return Convolution2D(filters, xdim, ydim,
                            activation=activation,
                            border_mode=border_mode,
                            init="he_uniform",
                            **kwargs)(model)

def conv(model, filters, dim=3, **kwargs):
    return convxy(model, filters, dim, dim, **kwargs)

def conv1d(model, filters, dim=3, **kwargs):
    model = convxy(model, filters, dim, 1, **kwargs)
    return convxy(model, filters, 1, dim, **kwargs)

def mpool(model, dim=2, **kwargs):
    return MaxPooling2D(pool_size=(dim, dim),
                           **kwargs)(model)

def apool(model, dim=2, **kwargs):
    return AveragePooling2D(pool_size=(dim, dim),
                           **kwargs)(model)

def gmpool(model):
    return GlobalMaxPooling2D()(model)

def gapool(model):
    return GlobalAveragePooling2D()(model)

def flatten(model, **kwargs):
    return Flatten(**kwargs)(model)

def dense(model, filters,
          activation='relu', **kwargs):
    return Dense(filters,
                    activation=activation,
                    init='he_uniform',
                    **kwargs)(model)

def finaldense(model, **kwargs):
    return dense(model, 1, activation='sigmoid', **kwargs)

def dropout(model, pct, **kwargs):
    return Dropout(pct, **kwargs)(model)

def activation(model, mode, **kwargs):
    return Activation(mode, **kwargs)(model)

def bottleneck(model, filters, dim=3, **kwargs):
    model = conv(model, filters//2, dim=1)
    model = conv(model, filters//2, dim=dim)
    return conv(model, filters,   dim=1)

def finalavg(model, **kwargs):
    model = conv(model, 1, dim=1, activation=None)
    model = gapool(model)
    model = flatten(model)
    return activation(model, 'sigmoid')

def finalmax(model, **kwargs):
    model = conv(model, 1, dim=1, activation=None)
    model = gmpool(model)
    model = flatten(model)
    return activation(model, 'sigmoid')

def bnorm(model, weight_decay=1E-4, **kwargs):
    return BatchNormalization(mode=0,
                              gamma_regularizer=l2(weight_decay),
                              beta_regularizer=l2(weight_decay),
                              **kwargs)(model)

def dn_conv(model, filters, weight_decay=1E-4, **kwargs):
    return conv(model, filters,
                activation=None, bias=False,
                W_regularizer=l2(weight_decay), **kwargs)

def dn_convstack(model, filters, bottleneck=None):
    model = activation(model, 'relu')
    if bottleneck and (bottleneck < model.get_shape()[-1]):
        model = dn_conv(model, bottleneck, dim=1)
        model = bnorm(model)
        model = activation(model, 'relu')
    return dn_conv(model, filters)

def dn_trans(model, filters):
    model = dn_conv(model, filters, dim=1)
    model = apool(model, dim=2)
    return bnorm(model)

def dn_dense(model, layers, in_filters, filters_per_layer, bottleneck=None):
    filters = in_filters
    flist = [model]
    for i in range(layers):
        model = dn_convstack(model, filters_per_layer, bottleneck=bottleneck)
        flist.append(model)
        model = merge(flist, mode='concat')
        filters += filters_per_layer
    return model, filters

def model_setup(model_name, learn_rate):
    model = None
    try:
        model = load_model("trained/"+model_name+".h5", custom_objects={'my_loss_fn':my_loss_fn})
        K.set_value(model.optimizer.lr, learn_rate)
        return model
    except Exception:
        pass

    # define the architecture of the network
    #model = Sequential()
    # builder = __import(model_name+".py")
    # exec(open(model_name+".py").read())
    builder = import_module(model_name)
    i, o = builder.build()
    model = Model(input=i, output=o)

    print("[INFO] compiling model...")
    model.compile(
            loss=my_loss_fn,
            optimizer=Adam(lr=learn_rate),
        metrics=["binary_accuracy"])

    return model

def model_checkpoint(model_name):
    return ModelCheckpoint("trained/"+model_name+".h5", save_best_only=True)

def model_export(model, model_name):
    # K.set_learning_phase(0)
    config = model.get_config()
    weights = model.get_weights()
    K.clear_session()
    K.set_learning_phase(0)
    model = Sequential.from_config(config)
    #model = model_from_config(config)
    # K.set_learning_phase(0)
    model.set_weights(weights)
    # saver = Saver()
    # saver.save(K.get_session(), "tf_checkpoint")
    graph_def = K.get_session().graph.as_graph_def()
    frozen_graph = convert_variables_to_constants(K.get_session(), graph_def, [model.output.name[:-2]])
    # opt_graph = optimize_for_inference(frozen_graph, [model.input.name[:-2]], [model.output.name[:-2]], tf.float32.as_datatype_enum)
    opt_graph = frozen_graph
    tf.reset_default_graph()
    tf.import_graph_def(opt_graph, name="")
    # rewrite = GraphRewriter()
    write_graph(opt_graph, "./tfmodel/", model_name+'.pb', as_text=False)
    print([o.name for o in tf.get_default_graph().get_operations()])
    # with open("tfnet.pb", "w") as f:
        # 1
        # f.write(str(graph_def))
