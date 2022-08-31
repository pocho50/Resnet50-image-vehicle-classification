from utils.data_aug import create_data_aug_layer
from tensorflow import keras, float32
from tensorflow.keras import layers, models, regularizers


def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = None,
    regulizer: float = 0.001,
):
    """
    Parameters
    ----------
    weights : str
        One of None (random initialization),
        'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.

    input_shape	: tuple
        Model input image shape as (height, width, channels).
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the input shape defined and we shouldn't change it.
        Input image size cannot be no smaller than 32. E.g. (224, 224, 3)
        would be one valid value.

    dropout_rate : float
        Value used for Dropout layer to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    data_aug_layer : dict
        Configuration from experiment YAML file used to setup the data
        augmentation process during finetuning.
        Only needed when weights='imagenet'.

    classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    Returns
    -------
    model : keras.Model
        Loaded model either ready for performing finetuning or to start doing
        predictions.
    """

    # Create the model to be used for finetuning here!
    if weights == "imagenet":
        # Define the Input layer
        # Assign it to `input` variable
        # Use keras.layers.Input(), following this requirements:
        #   1. layer dtype must be tensorflow.float32
        input = layers.Input(shape=input_shape, dtype=float32)

        # Create the data augmentation layers here and add to the model next
        # to the input layer
        # If no data augmentation was used, skip this
        if data_aug_layer is not None:
            data_augmentation = create_data_aug_layer(data_aug_layer)
            x = data_augmentation(input)
        else:
            x = input

        # Add a layer for preprocessing the input images values
        # E.g. change pixels interval from [0, 255] to [0, 1]
        # Resnet50 already has a preprocessing function you must use here
        # See keras.applications.resnet50.preprocess_input()
        x = keras.applications.resnet50.preprocess_input(x)

        # Create the corresponding core model using
        # keras.applications.ResNet50()
        # The model created here must follow this requirements:
        #   1. Use imagenet weights
        #   2. Drop top layer (imagenet classification layer)
        #   3. Use Global average pooling as model output
        base_model = keras.applications.ResNet50(
            include_top=False, pooling="avg", weights="imagenet"
        )

        x = base_model(x)

        # Add a single dropout layer for regularization, use
        # keras.layers.Dropout()
        x = layers.Dropout(dropout_rate)(x)

        # Add the classification layer here, use keras.layers.Dense() and
        # `classes` parameter
        # Assign it to `outputs` variable
        outputs = layers.Dense(
            classes, activation="softmax", kernel_regularizer=regularizers.L2(regulizer)
        )(x)

        # Now you have all the layers in place, create a new model
        # Use keras.Model()
        # Assign it to `model` variable
        model = keras.Model(input, outputs)

    else:

        model = models.load_model(weights)

    return model
