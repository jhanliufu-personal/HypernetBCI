import torch
from torch.nn.utils.stateless import functional_call
# from models.Embedder import Conv1dEmbedder
# from models.Hypernet import LinearHypernet
# from Embedder import Conv1dEmbedder
# from Hypernet import LinearHypernet


class HyperBCINet(torch.nn.Module):
    """
    A HyperBCINet is built on top of an existing primary network (say ShallowFBCSP net) PLUS
    an embedder and a hypernet.

    HyperBCINet can be in one of three states: TRAINING, CALIBRATION and TESTING. 
    Only TRAINING stage involves backprop; it trains the primary network, embedder and hypernet. 
    In CALIBRATION stage, the trained embedder and hypernet are used to generate weights / calibrate. 
    In TESTING stage, no weight generation / backprop whatsoever. Inference using existing weights.

    For training, when a batch of input data samples are given, we will get a batch of embeddings and 
    weight tensors. We can either AGGREGATE the weight tensors to get one weight tensor for the whole
    batch, or evaluate each input sample with its own weight tensor.
    """
    def __init__(
            self, 
            primary_net: torch.nn.Module,
            embedder:  torch.nn.Module,
            embedding_shape: torch.Size, 
            sample_shape: torch.Size,
            hypernet: torch.nn.Module
        ) -> None:
        """
        Parameters
        ---------------------------------------
        primary_net: nn.Module, the model that hypernet is built upon
        embedding_shape: torch.Size, shape of the embedding. (embedding_dimension, embedding_length)
        sample_shape: torch.Size, shape of input data samples
        """
        super(HyperBCINet, self).__init__()

        # The primary network and its parameters
        self.primary_net = primary_net
        # need this for functional forward call
        self.primary_params = {name: param for name, param in primary_net.named_parameters()}

        self.sample_shape = sample_shape
        self.embedding_shape = embedding_shape

        ### ----------------------------------------------------------------------
        """
        Eventually, this part should be fully generic. Any number of specified model
        layers can have an HN, and the embedder and HN(s) can be any child of the
        Embedder and Hypernet parent classes.
        """

        # for simplicity, only have one hypernet for now on conv_classifier
        self.weight_shape = primary_net.final_layer.conv_classifier.weight.shape

        # embedder
        # self.embedder = Conv1dEmbedder(sample_shape, embedding_shape)
        self.embedder = embedder

        # hypernet / weight generator
        # self.hypernet = LinearHypernet(embedding_shape, self.weight_shape)
        self.hypernet = hypernet
        ### ----------------------------------------------------------------------

        self.calibrating = False

    def calibrate(self) -> None:
        self.calibrating = True

    def aggregate_tensors(self, tensors: torch.Tensor, aggr='Avg') -> torch.Tensor:
        """
        Aggregate a tensor (a batch of tensors) along its first dimension. If the input
        tensor has shape [a, b, c], output tensor will have shape [b, c]

        Parameters
        ---------------------------------------
        tensors: torch.Tensor, a batch of tensors.
        aggr: str, aggregation method
        """ 
        match aggr:
            case 'Avg':
                aggregated_tensor = torch.mean(tensors, dim=0)
            case _:
                raise ValueError('Aggregation method is not defined.')

        return aggregated_tensor

    def forward(self, x, aggr='Avg', random_update=False):
        """
        Parameters
        ---------------------------------------
        x: torch tensor, has shape: (batch_size, *sample_shape)
        """
        assert x.shape[1:] == self.sample_shape, "Input has incorrect shape"

        # For model training and calibration, generate embedding and new weight tensor
        if self.training or self.calibrating:

            # Output random weight tensor as control
            # This would detach the embedder and weight generator from
            # the computation graph. backprop won't reach them.
            if random_update:
                print('Update weight tensor to RANDOM TENSOR')
                random_weight_tensor = torch.randn(self.weight_shape).cuda()
                # random_weight_tensor.cuda()
                self.primary_params.update({'final_layer.conv_classifier.weight': random_weight_tensor})
                print('Functional call using RANDOM WEIGHT TENSOR')
                return functional_call(self.primary_net, self.primary_params, x)

            print('Generate new embedding and weights')
            # generate embeddings
            embeddings = self.embedder(x)
            # generate new weight tensors
            new_weight_tensors = torch.stack([self.hypernet(emb) for emb in embeddings])

            # aggregate the weight tensors if specified to
            if aggr is not None:
                print('Aggregate weight tensors')
                aggregated_weight_tensor = self.aggregate_tensors(new_weight_tensors, aggr=aggr)
                assert aggregated_weight_tensor.shape == self.weight_shape, "Weight tensor has incorrect shape"

                # update weights
                print('Update new tensor to model parameters')
                # self.primary_net.final_layer.conv_classifier.weight = nn.Parameter(aggregated_weight_tensor, requires_grad=False)
                self.primary_params.update({'final_layer.conv_classifier.weight': aggregated_weight_tensor})

            # else evaluate each input with its corresponding weight tensor
            else:
                assert not self.calibrating, "Must aggregate if in calibration mode"
                return None
    
        # print('Forward pass using functional call')
        return functional_call(self.primary_net, self.primary_params, x)