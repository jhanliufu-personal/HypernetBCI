from braindecode.models import ShallowFBCSPNet
from torch import Size, randn
from utils import parse_training_config
from models.HypernetBCI import HyperBCINet

args = parse_training_config()

n_channels = 22
n_classes = 4
input_window_samples = 2250
sample_shape = Size([n_channels, input_window_samples])
model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

# In Ha et.al 2016 the embedding is 1 dimensional
out_channels = 1
out_length = 749
embedding_shape = Size([out_channels, out_length])
weight_shape = model.final_layer.conv_classifier.weight.shape
myHNBCI = HyperBCINet(model, embedding_shape, sample_shape)

test_input = randn([3, n_channels, input_window_samples])
myHNBCI.calibrate()
myHNBCI(test_input, **(args.forward_pass_kwargs))