'''
This file implements models that involve three modules: 
A support encoder that gives embeddings as modal support (acquired thru contrastive learning)
An encoder that produces embeddings for classification
A classifier that takes embeddings + support embeddings
'''
import torch


class Supportnet(torch.nn.Module):
    def __init__(self, support_encoder, encoder, classifier) -> None:
        super(Supportnet, self).__init__() 
        self.support_encoder = support_encoder
        self.encoder = encoder
        self.classifier = classifier
        # self.batch_size = batch_size

    def concatenate_embeddings(self, support_emb, emb):
        # support_emb assumed to have shape [40, 144, 1]; reshaped to [40]
        support_emb = support_emb.squeeze(-1)[:,:,-1]
        cur_batch_size = support_emb.size(0)
        support_emb_broadcasted = support_emb.view(cur_batch_size, 40, 1, 1).expand(-1, -1, 144, 1)
        # the concated embedding should have shape [80, 144, 1]
        return torch.cat((emb, support_emb_broadcasted), dim=1)
    
    def forward(self, x):
        _ = self.support_encoder(x)
        support_embedding = self.support_encoder.get_embeddings()
        _ = self.encoder(x)
        embedding = self.encoder.get_embeddings()

        concatenated_embedding = self.concatenate_embeddings(support_embedding, embedding)
        return self.classifier(concatenated_embedding).squeeze(-1).squeeze(-1)

