'''
Reproduced from Ozyurt et.al 2023 (CLUDA)
'''
import torch
from torch import nn
from braindecode.models import ShallowFBCSPNet


'''
MLP for projector, predictor and adversarial domain discriminator
'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, use_batch_norm=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None

    def forward(self, x, static):
        
        if static is not None:
            hidden = self.input_fc(torch.cat([x, static], dim=1))
        else:
            hidden = self.input_fc(x)
        
        if self.use_batch_norm:
            hidden = self.batch_norm(hidden)

        hidden = nn.functional.relu(hidden)
        
        y_pred = self.output_fc(hidden)
        
        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)

        return y_pred


'''
Nearest-neighbor algorithm for finding the closest neighbors of target keys
in source queries
'''
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def NN(key, queue, num_neighbors=1, return_indices=False):
    """
    key: N x D matrix
    queue: M x D matrix
    
    output: num_neighbors x N x D matrix for closest neighbors of key within queue
    NOTE: Output is unnormalized
    """
    #Apply Cosine similarity (equivalent to l2-normalization + dot product)
    similarity = sim_matrix(key, queue)
    
    indices_top_neighbors = torch.topk(similarity, k=num_neighbors, dim=1)[1]
    
    list_top_neighbors = []
    for i in range(num_neighbors):
        indices_ith_neighbor = indices_top_neighbors[:,i]
        list_top_neighbors.append(queue[indices_ith_neighbor,:])
        
    if return_indices:
        return torch.stack(list_top_neighbors), indices_top_neighbors
    else:
        return torch.stack(list_top_neighbors)
    

class ShallowFBCSPEncoder(nn.Module):
    def __init__(
            self, 
            sample_shape: torch.Size, 
            layer_name: str,
            n_classes: int
        ) -> None:
        super(ShallowFBCSPEncoder, self).__init__()  
        self.output = None
        # self.embedding_shape = embedding_shape
        self.model = ShallowFBCSPNet(
            sample_shape[0],
            n_classes,
            input_window_samples=sample_shape[1],
            final_conv_length="auto"
        )
        self.layer_name = layer_name
        self.hook = getattr(self.model, layer_name).register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def forward(self, x):
        _ = self.model(x)
        # assert self.output.shape[1:] == self.embedding_shape, (
        #     f'output embedding has wrong shape ({self.output.shape[1:]}) ' +
        #     f'correct embedding shape is {self.embedding_shape}'
        # )
        return self.output
    
    def close_hook(self):
        self.hook.remove()


'''
Taken from the original TC implementation https://github.com/locuslab/TCN
'''
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, stride=1, dilation_factor=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilation_factor ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)