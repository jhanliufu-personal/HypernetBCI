import torch
from torch.nn.utils.stateless import functional_call
from torch.nn.functional import softmax
from copy import deepcopy
from models.Hypernet import LinearHypernet

class HyperProtoNet(torch.nn.Module):

    def __init__(
        self, 
        primary_net: torch.nn.Module,
        embedding_shape: torch.Size,
        num_classes = 4,
        device = 'cuda'
    ):
        super().__init__()

        self.encoder = torch.nn.Sequential(*list(primary_net.children())[:-1])

        self.num_classes = num_classes
        self.embedding_shape = embedding_shape
        self.class_prototypes_shape = torch.Size([
            self.num_classes, *self.embedding_shape
        ])

        self.flattened_dim = int(torch.prod(torch.tensor(embedding_shape)).item())
        self.clf_weight_shape = torch.Size([self.num_classes, self.flattened_dim])

        # self.hypernet = LinearHypernet(
        #     self.embedding_shape, self.clf_weight_shape[1:]
        # )
        # self.hypernet = LinearHypernet(
        #     self.class_prototypes_shape, self.clf_weight_shape
        # )


    def calculate_class_proto(self, proto_x, proto_y, device='cuda'):
        '''
        (proto_x, proto_y) = labeled support set i.e prototype set
        '''
        proto_emb = self.encoder(proto_x)
        proto_emb = proto_emb.squeeze(-1)

        class_prototypes = torch.zeros(
            (self.num_classes, *self.embedding_shape),
            device = device
        )
        for c in range(self.num_classes):
            class_mask = (proto_y == c)
            if class_mask.sum() > 0:
                class_prototypes[c] = proto_emb[class_mask].mean(dim=0)

        return class_prototypes


    def update_clf_weights(self, proto_x, proto_y, device='cuda'):

        class_prototypes = self.calculate_class_proto(proto_x, proto_y)
        # clf_weight = torch.zeros(*self.clf_weight_shape, device = device)

        '''
        We can either have the HN do 
            class_prototypes -> weights,
        or
            class_prototype[i] -> weights[i]
        '''
        # for i in range(self.num_classes):
        #     clf_weight[i] = self.hypernet(class_prototypes[i])

        # clf_weight = self.hypernet(class_prototypes)

        clf_weight = class_prototypes.view(*self.clf_weight_shape)

        return clf_weight


    def forward(self, query_x, proto_x, proto_y, device='cuda'):
        # Update classifier weights
        clf_weight = self.update_clf_weights(proto_x, proto_y, device = device)

        # Forward pass of query samples
        query_emb = self.encoder(query_x)
        # shape: [B, 5760]
        query_emb = query_emb.squeeze(-1).flatten(start_dim=1)  

        logits = torch.matmul(query_emb, clf_weight.T)
        out = softmax(logits, dim=1)

        return out


        

