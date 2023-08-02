from keras import Model
from keras import layers
from keras import applications

"""
Models for butter package
"""

def createNewButterModelv1(X_conv):
    ################################################################
    # Build Model - Base model
    ################################################################
    base_model = applications.MobileNetV2(input_shape=X_conv.shape[1:], include_top=False, weights='imagenet')
    base_model.trainable = False

    ################################################################
    # Build Model - Linker model
    ################################################################
    l2_MaxP = layers.MaxPooling2D(pool_size=(3, 3))(base_model.get_layer('block_2_add').output)
    l5_ConvT = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        base_model.get_layer('block_5_add').output)
    l5_MaxP = layers.MaxPooling2D(pool_size=(3, 3))(l5_ConvT)
    l12_ConvT = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(4, 4), padding='valid', output_padding=(1, 1))(
        base_model.get_layer('block_12_add').output)
    l12_MaxP = layers.MaxPooling2D(pool_size=(3, 3), padding='valid')(l12_ConvT)
    linker_input = layers.concatenate([l2_MaxP, l5_MaxP, l12_MaxP])
    linker_output = layers.Flatten()(linker_input)

    ################################################################
    # Build Model - FC model
    ################################################################
    FC = layers.Dropout(0.2, name='FC_DO1')(linker_output)
    FC = layers.Dense(500, activation='relu', name='FC_1')(FC)
    FC = layers.Dropout(0.2, name='FC_DO2')(FC)
    FC = layers.Dense(200, activation='relu', name='FC_2')(FC)
    FC = layers.Dropout(0.1, name='FC_DO3')(FC)
    FC = layers.Dense(4, activation='linear', name='FC_3')(FC)

    ################################################################
    # Compile and Train
    ################################################################

    new_model = Model(inputs=base_model.input, outputs=FC)

    return new_model
