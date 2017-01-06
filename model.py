from keras.models import Sequential, load_model, save_model, model_from_config
from keras.layers import Activation
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import _to_tensor, _EPSILON
from keras import backend as K
import tensorflow as tf
from tensorflow.python.training.training import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
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

def convxy(model, layers, xdim, ydim, **kwargs):
    model.add(Convolution2D(layers, xdim, ydim,
                            activation="relu",
                            border_mode="same",
                            init="he_normal",
                            **kwargs))

def conv(model, layers, dim=3, **kwargs):
    convxy(model, layers, dim, dim, **kwargs);

def conv1d(model, layers, dim=3, **kwargs):
    convxy(model, layers, dim, 1, **kwargs);
    convxy(model, layers, 1, dim, **kwargs);

def pool(model, dim=2, **kwargs):
    model.add(MaxPooling2D(pool_size=(dim, dim),
                           **kwargs))

def flatten(model, **kwargs):
    model.add(Flatten(**kwargs))

def dense(model, layers, **kwargs):
    model.add(Dense(layers,
                    activation="relu",
                    init='he_normal',
                    **kwargs))

def finaldense(model, **kwargs):
    model.add(Dense(1,
                    activation="sigmoid",
                    init='he_normal',
                    **kwargs))

def dropout(model, pct, **kwargs):
    model.add(Dropout(pct, **kwargs))

def bottleneck(model, layers):
    conv(model, layers/2, dim=1)
    conv(model, layers/2)
    conv(model, layers, dim=1)

def finalavg(model, **kwargs):
    model.add(Convolution2D(1, 1, 1, activation=None, init='he_normal'))
    dim = model.output_shape[1]
    model.add(AveragePooling2D((dim, dim)))
    flatten(model)
    model.add(Activation('sigmoid'))

def finalmax(model, **kwargs):
    model.add(Convolution2D(1, 1, 1, activation=None, init='he_normal'))
    dim = model.output_shape[1]
    model.add(MaxPooling2D((dim, dim)))
    flatten(model)
    model.add(Activation('sigmoid'))

def model_setup(model_name):
    model = None
    try:
        model = load_model("trained/"+model_name+".h5", custom_objects={'my_loss_fn':my_loss_fn})
        return model
    except Exception:
        pass

    # define the architecture of the network
    model = Sequential()
    exec(open(model_name+".py").read())

    print("[INFO] compiling model...")
    model.compile(
            loss=my_loss_fn,
            optimizer='adam',
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
    # K.set_learning_phase(0)
    model.set_weights(weights)
    # saver = Saver()
    # saver.save(K.get_session(), "tf_checkpoint")
    graph_def = K.get_session().graph.as_graph_def()
    frozen_graph = convert_variables_to_constants(K.get_session(), graph_def, [model.output.name[:-2]])
    opt_graph = optimize_for_inference(frozen_graph, [model.input.name[:-2]], [model.output.name[:-2]], tf.float32.as_datatype_enum)
    tf.reset_default_graph()
    tf.import_graph_def(opt_graph, name="")
    # rewrite = GraphRewriter()
    write_graph(opt_graph, "./tfmodel/", model_name+'.pb', as_text=False)
    print([o.name for o in tf.get_default_graph().get_operations()])
    # with open("tfnet.pb", "w") as f:
        # 1
        # f.write(str(graph_def))
