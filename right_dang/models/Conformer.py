from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization 
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential

from einops.layers.tensorflow import Rearrange


def PatchEmbedding(num_channel, emb_size):
    return Sequential([
        # Shallow Net
        Conv2D(40, (1, 25), (1, 1)),
        Conv2D(40, (22, 1), (1, 1)),
        BatchNormalization(),
        Activation('elu'),
        AveragePooling2D((1, 75), (1, 15)),
        Dropout(0.5),

        # Projection
        Conv2D(emb_size, (1, 1), (1, 1)),
        Rearrange('b e (h) (w) -> b (h w) e')
    ])

     
print(PatchEmbedding(4, 40))