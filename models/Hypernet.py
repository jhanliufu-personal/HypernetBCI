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
    def __init__(self, embedding_shape: torch.Size, weight_shape: torch.Size) -> None:
        super(LinearHypernet, self).__init__(embedding_shape, weight_shape)
        # Initialize weight generation layer
        self.input_size = embedding_shape[0] * embedding_shape[1]
        self.output_size = torch.prod(torch.tensor(weight_shape)).item()
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

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
        # Check embedding shape
        if embedding.shape != self.embedding_shape:
            raise ValueError('Embedding has wrong shape')

        # Flatten the input tensor
        embedding_flattened = embedding.view(-1, self.input_size)
        # Apply the fully connected layer
        weight_flattened = self.fc(embedding_flattened)
        # Reshape the output to the desired weight shape
        weight_reshaped = weight_flattened.view(*(self.weight_shape))

        # Check weight tensor shape
        if weight_reshaped.shape != self.weight_shape:
            raise ValueError('The generated weight tensor has wrong shape')

        return weight_reshaped