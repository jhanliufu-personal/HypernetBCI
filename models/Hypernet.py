import torch


class Hypernet(torch.nn.Module):
    """
    A Hypernetwork takes one embedding and generates a weight tensor
    """
    def __init__(self, embedding_shape: torch.Size, weight_shape: torch.Size) -> None:
        """
        embedding_shape: (embedding_dimension, embedding_length)
        weight_shape: arbitrary
        """
        super(Hypernet, self).__init__()
        self.embedding_shape = embedding_shape
        self.weight_shape = weight_shape

    def forward(self, embedding):
        """
        Parameters
        ---------------------------------------
        embedding: embedding for one input data sample, has
        shape: embedding_shape

        return
        ---------------------------------------
        weight_tensor: weight tensor generated from one
        embedding. has shape: weight_shape
        """
        raise NotImplementedError("Each hypernet must implement the forward method.")


class LinearHypernet(Hypernet):
    def __init__(
            self, 
            input_shape: torch.Size, 
            output_shape: torch.Size
        ) -> None:
        super(LinearHypernet, self).__init__(input_shape, output_shape)
        # Initialize weight generation layer
        self.input_shape = input_shape
        self.input_size = torch.prod(torch.tensor(input_shape)).item()

        self.output_shape = output_shape
        self.output_size = torch.prod(torch.tensor(output_shape)).item()

        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        # Check input shape
        assert x.shape == self.input_shape, (
            'LinearHypernet.forward(): '
            'Input has wrong shape'
        )
        
        # print(x.is_cuda)

        # Flatten the input tensor
        x_flattened = x.view(-1, self.input_size)
        # Apply the fully connected layer
        out_flattened = self.fc(x_flattened)
        # Reshape the output to the desired weight shape
        out_reshaped = out_flattened.view(*(self.output_shape))

        # Check weight tensor shape
        assert out_reshaped.shape == self.output_shape, (
            'LinearHypernet.forward(): Output has wrong shape'
        )

        return out_reshaped