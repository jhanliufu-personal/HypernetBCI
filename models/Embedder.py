import torch
from braindecode.models import ShallowFBCSPNet


class Embedder(torch.nn.Module):
    """
    An embedder takes a batch of input data samples and returns a batch of
    embeddings, one for each input sample
    """
    def __init__(self, sample_shape: torch.Size, embedding_shape: torch.Size) -> None:
        """
        sample_shape: (in_dimension, in_length)
        embedding_shape: (embedding_dimension, embedding_length)
        """
        super(Embedder, self).__init__()
        self.sample_shape = sample_shape
        # self.in_dimension = sample_shape[0]
        # self.in_length = sample_shape[1]
        self.embedding_shape = embedding_shape
        # self.embedding_dimension = embedding_shape[0]
        # self.embedding_length = embedding_shape[1]

    def forward(self, x):
        """
        Parameters
        ---------------------------------------
        x: torch tensor, has shape: (batch_size, *sample_shape)

        return
        ---------------------------------------
        embeddings: embeddings for each input data sample, has
        shape: (batch_size, *embedding_shape)
        """
        raise NotImplementedError("Each embedder must implement the forward method.")


class ShallowFBCSPEmbedder(Embedder):
    def __init__(
            self, 
            sample_shape: torch.Size, 
            embedding_shape: torch.Size, 
            layer_name: str,
            n_classes: int
        ) -> None:
        super(ShallowFBCSPEmbedder, self).__init__(sample_shape, embedding_shape)  
        self.output = None
        self.embedding_shape = embedding_shape
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
        assert (
            self.output.shape[1:] == self.embedding_shape, 
            f'output embedding has wrong shape ({self.output.shape[1:]}) ' +
            f'correct embedding shape is {self.embedding_shape}'
        )
        return self.output
    
    def close_hook(self):
        self.hook.remove()


class Conv1dEmbedder(Embedder):
    def __init__(
            self, 
            sample_shape: torch.Size,
            embedding_shape: torch.Size, 
            kernel_size=5, 
            stride=3, 
            padding=0, 
            dilation=1
        ) -> None:
        super(Conv1dEmbedder, self).__init__(sample_shape, embedding_shape)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Initialize 1D convolution
        self.conv = torch.nn.Conv1d(
            self.sample_shape[0], 
            self.embedding_shape[0], 
            self.kernel_size, 
            self.stride, 
            self.padding
        )

    def forward(self, x):
        """
        Parameters
        ---------------------------------------
        x: torch tensor, has shape: (batch_size, *sample_shape)

        return
        ---------------------------------------
        embeddings: embeddings for each input data sample, has
        shape: (batch_size, *embedding_shape)
        """
        embeddings = self.conv(x)
        return embeddings