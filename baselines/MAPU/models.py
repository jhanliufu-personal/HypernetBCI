import torch
from torch import nn
from einops import rearrange
from braindecode.models import ShallowFBCSPNet


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]

## Feature Extractor
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                configs.input_channels, 
                configs.mid_channels, 
                kernel_size=configs.kernel_size,
                stride=configs.stride, 
                bias=False, 
                padding=(configs.kernel_size // 2)
            ),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2, 
                stride=2, 
                padding=1
            ),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                configs.mid_channels, 
                configs.mid_channels * 2, 
                kernel_size=8, 
                stride=1, 
                bias=False, 
                padding=4
            ),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2, 
                stride=2, 
                padding=1
            )
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                configs.mid_channels * 2, 
                configs.final_out_channels, 
                kernel_size=8, 
                stride=1, 
                bias=False,
                padding=4
            ),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = self.aap(x).view(x.shape[0], -1)

        return x_flat, x

class ShallowFBCSPFeatureExtractor(torch.nn.Module):
    def __init__(
            self, 
            sample_shape: torch.Size, 
            layer_name: str,
            n_classes: int
        ) -> None:
        super(ShallowFBCSPFeatureExtractor, self).__init__()  
        self.features = None  # To store the output of the hooked layer
        self.logits = None    # To store the logits from the conv_classifier layer
        
        self.model = ShallowFBCSPNet(
            sample_shape[0],
            n_classes,
            input_window_samples=sample_shape[1],
            final_conv_length="auto"
        )
        
        self.layer_name = layer_name
        # Hook to capture the specified layer's output
        self.hook = getattr(self.model, layer_name).register_forward_hook(self.hook_fn)
        # Hook to capture the output of the conv_classifier layer
        self.logits_hook = self.model.final_layer.conv_classifier.register_forward_hook(self.logits_hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def logits_hook_fn(self, module, input, output):
        self.logits = output

    def forward(self, x):
        prediction = self.model(x)
        return self.features, prediction

    def close_hooks(self):
        self.hook.remove()
        self.logits_hook.remove()

##  Classifier
class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions
    
## Temporal Imputer
class Temporal_Imputer(nn.Module):
    def __init__(self, configs):
        super(Temporal_Imputer, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        out = out.view(x.size(0), self.num_channels, -1)
        # take the last time step
        return out

## not using the config object
class myTemporal_Imputer(torch.nn.Module):
    # hidden dimension should be the same as final out channels
    def __init__(self, final_out_channels, AR_hid_dim):
        super(myTemporal_Imputer, self).__init__()
        self.num_channels = final_out_channels
        self.hid_dim = AR_hid_dim
        self.rnn = torch.nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        out = out.view(x.size(0), self.num_channels, -1)
        # take the last time step
        return out

# temporal masking
def masking(x, num_splits=8, num_masked=4):
    # num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masks = []
    for i in range(masked_patches.shape[1]):
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
        # orig_patches[:, i, (selected_indices[i, :]), :] =
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask