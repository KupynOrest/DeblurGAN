import copy
import numpy as np

from keras import backend as K
from keras.engine.training import objectives, standardize_input_data, slice_X, \
    standardize_sample_weights, standardize_class_weights, standardize_weights, check_loss_and_target_compatibility


def smooth_gan_labels(y):
    assert len(y.shape) == 2, "Needs to be a binary class"
    y = np.asarray(y, dtype='int')
    Y = np.zeros(y.shape, dtype='float32')

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] == 0:
                Y[i, j] = np.random.uniform(0.0, 0.3)
            else:
                Y[i, j] = np.random.uniform(0.7, 1.2)

    return Y


def _standardize_user_data(model, x, y,
                           sample_weight=None, class_weight=None,
                           check_batch_dim=True, batch_size=None):
    if not hasattr(model, 'optimizer'):
        raise Exception('You must compile a model before training/testing.'
                        ' Use `model.compile(optimizer, loss)`.')

    output_shapes = []
    for output_shape, loss_fn in zip(model.internal_output_shapes, model.loss_functions):
        if loss_fn.__name__ == 'sparse_categorical_crossentropy':
            output_shapes.append(output_shape[:-1] + (1,))
        elif getattr(objectives, loss_fn.__name__, None) is None:
            output_shapes.append(None)
        else:
            output_shapes.append(output_shape)
    x = standardize_input_data(x, model.input_names,
                               model.internal_input_shapes,
                               exception_prefix='model input')
    y = standardize_input_data(y, model.output_names,
                               output_shapes,
                               exception_prefix='model target')
    sample_weights = standardize_sample_weights(sample_weight,
                                                model.output_names)
    class_weights = standardize_class_weights(class_weight,
                                              model.output_names)
    sample_weights = [standardize_weights(ref, sw, cw, mode)
                      for (ref, sw, cw, mode)
                      in zip(y, sample_weights, class_weights, model.sample_weight_modes)]

    '''
    We only need to comment out check_array_lengeh(x, y, weights) in the next line to
    let the model compile and train.
    '''
    # check_array_lengths(x, y, sample_weights)

    check_loss_and_target_compatibility(y, model.loss_functions, model.internal_output_shapes)
    if model.stateful and batch_size:
        if x[0].shape[0] % batch_size != 0:
            raise Exception('In a stateful network, '
                            'you should only pass inputs with '
                            'a number of samples that can be '
                            'divided by the batch size. Found: ' +
                            str(x[0].shape[0]) + ' samples')
    return x, y, sample_weights


def fit(model, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
        validation_split=0., validation_data=None, shuffle=True,
        class_weight=None, sample_weight=None):
    '''Trains the model for a fixed number of epochs (iterations on a dataset).

    # Arguments
        x: Numpy array of training data,
            or list of Numpy arrays if the model has multiple inputs.
            If all inputs in the model are named, you can also pass a dictionary
            mapping input names to Numpy arrays.
        y: Numpy array of target data,
            or list of Numpy arrays if the model has multiple outputs.
            If all outputs in the model are named, you can also pass a dictionary
            mapping output names to Numpy arrays.
        batch_size: integer. Number of samples per gradient update.
        nb_epoch: integer, the number of times to iterate over the training data arrays.
        verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose, 2 = one log line per epoch.
        callbacks: list of callbacks to be called during training.
            See [callbacks](/callbacks).
        validation_split: float between 0 and 1:
            fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate the loss and any model metrics
            on this data at the end of each epoch.
        validation_data: data on which to evaluate the loss and any model metrics
            at the end of each epoch. The model will not be trained on this data.
            This could be a tuple (x_val, y_val) or a tuple (val_x, val_y, val_sample_weights).
        shuffle: boolean, whether to shuffle the training data before each epoch.
        class_weight: optional dictionary mapping class indices (integers) to
            a weight (float) to apply to the model's loss for the samples
            from this class during training.
            This can be useful to tell the model to "pay more attention" to
            samples from an under-represented class.
        sample_weight: optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify sample_weight_mode="temporal" in compile().


    # Returns
        A `History` instance. Its `history` attribute contains
        all information collected during training.
    '''
    # validate user data
    '''
    We need to use the custom standardize_user_data function defined above
    to skip checking for the array_length. Everything else is the same as the original fit code

    Note: We did not use model._standardize_user_data(...) but instead used the above
    standardize_user_data(...) method to bypass the original standardize code in keras.
    '''
    x, y, sample_weights = _standardize_user_data(model, x, y,
                                                  sample_weight=sample_weight,
                                                  class_weight=class_weight,
                                                  check_batch_dim=False,
                                                  batch_size=batch_size)
    # prepare validation data
    if validation_data:
        do_validation = True
        if len(validation_data) == 2:
            val_x, val_y = validation_data
            val_sample_weight = None
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data
        else:
            raise
        val_x, val_y, val_sample_weights = model._standardize_user_data(val_x, val_y,
                                                                        sample_weight=val_sample_weight,
                                                                        check_batch_dim=False,
                                                                        batch_size=batch_size)
        model._make_test_function()
        val_f = model.test_function
        if model.uses_learning_phase and type(K.learning_phase()) is not int:
            val_ins = val_x + val_y + val_sample_weights + [0.]
        else:
            val_ins = val_x + val_y + val_sample_weights

    elif validation_split and 0. < validation_split < 1.:
        do_validation = True
        split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
        y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
        sample_weights, val_sample_weights = (
            slice_X(sample_weights, 0, split_at), slice_X(sample_weights, split_at))
        model._make_test_function()
        val_f = model.test_function
        if model.uses_learning_phase and type(K.learning_phase()) is not int:
            val_ins = val_x + val_y + val_sample_weights + [0.]
        else:
            val_ins = val_x + val_y + val_sample_weights
    else:
        do_validation = False
        val_f = None
        val_ins = None

    # prepare input arrays and training function
    if model.uses_learning_phase and type(K.learning_phase()) is not int:
        ins = x + y + sample_weights + [1.]
    else:
        ins = x + y + sample_weights
    model._make_train_function()
    f = model.train_function

    # prepare display labels
    out_labels = model.metrics_names

    # rename duplicated metrics name
    # (can happen with an output layer shared among multiple dataflows)
    deduped_out_labels = []
    for i, label in enumerate(out_labels):
        new_label = label
        if out_labels.count(label) > 1:
            dup_idx = out_labels[:i].count(label)
            new_label += '_' + str(dup_idx + 1)
        deduped_out_labels.append(new_label)
    out_labels = deduped_out_labels

    if do_validation:
        callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
    else:
        callback_metrics = copy.copy(out_labels)

    # delegate logic to _fit_loop
    return model._fit_loop(f, ins, out_labels=out_labels,
                           batch_size=batch_size, nb_epoch=nb_epoch,
                           verbose=verbose, callbacks=callbacks,
                           val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                           callback_metrics=callback_metrics)
