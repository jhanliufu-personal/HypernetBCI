'''
This file implements models that involve three modules: 
A support encoder that gives embeddings as modal support (acquired thru contrastive learning)
An encoder that produces embeddings for classification
A classifier that takes embeddings + support embeddings
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Supportnet(torch.nn.Module):
    def __init__(self, support_encoder, encoder, classifier) -> None:
        super(Supportnet, self).__init__() 
        self.support_encoder = support_encoder
        self.encoder = encoder
        self.classifier = classifier
        # self.batch_size = batch_size

        # Linear transformation for attention mechanism
        self.query_layer = nn.Linear(40, 40)  # Query from task embedding (per time step)
        self.key_layer = nn.Linear(40, 40)    # Key from support embedding
        self.value_layer = nn.Linear(40, 40)  # Value from task embedding (per time step)

        self.integrated_embeddings = None

    def concatenate_embeddings(self, support_emb, emb):
        # support_emb assumed to have shape [40, 144, 1]; reshaped to [40]
        support_emb = support_emb.squeeze(-1)[:,:,-1]
        cur_batch_size = support_emb.size(0)
        support_emb_broadcasted = support_emb.view(cur_batch_size, 40, 1, 1).expand(-1, -1, 144, 1)
        # the concated embedding should have shape [80, 144, 1]
        return torch.cat((emb, support_emb_broadcasted), dim=1)
    

    def attention_transform(self, support_emb, task_emb):
        """
        Apply attention to the task embedding based on the support embedding.
        support_emb has shape [batch_size, 40, 144, 1], but we take the last time step (shape [batch_size, 40])
        task_emb has shape [batch_size, 40, 144, 1]
        """
        # Remove the last singleton dimension from task_emb (size: [batch_size, 40, 144])
        task_emb = task_emb.squeeze(-1)

        # Take the last time step from the support embedding (shape: [batch_size, 40])
        support_emb_last_step = support_emb[:, :, -1].squeeze(-1)  # Shape: [batch_size, 40]

        # Compute queries, keys, and values using trainable layers
        # Shape: [batch_size, 144, 40]
        query = self.query_layer(task_emb.transpose(1, 2))  
        # Shape: [batch_size, 40]
        key = self.key_layer(support_emb_last_step)     
        # Shape: [batch_size, 144, 40]     
        value = self.value_layer(task_emb.transpose(1, 2))   

        # Compute attention scores (scaled dot product) between query and key
        # Shape: [batch_size, 144]
        attention_scores = torch.bmm(query, key.unsqueeze(-1)).squeeze(-1)
        # Shape: [batch_size, 144]
        attention_weights = F.softmax(attention_scores, dim=-1)             
        # Apply attention weights to the value (task embedding)
        # Shape: [batch_size, 144, 40]
        attended_value = value * attention_weights.unsqueeze(-1)  

        # Transpose back to [batch_size, 40, 144] and add singleton dimension for consistency
        # Shape: [batch_size, 40, 144, 1]
        transformed_task_emb = attended_value.transpose(1, 2).unsqueeze(-1)  

        # Add the transformed embedding back to the original task embedding (residual connection)
        # Shape: [batch_size, 40, 144, 1]
        updated_task_emb = transformed_task_emb + task_emb.unsqueeze(-1)  

        return updated_task_emb


    def forward(self, x):
        _ = self.support_encoder(x)
        support_embedding = self.support_encoder.get_embeddings()
        _ = self.encoder(x)
        embedding = self.encoder.get_embeddings()

        # concatenated_embedding = self.concatenate_embeddings(support_embedding, embedding)
        # return self.classifier(concatenated_embedding).squeeze(-1).squeeze(-1)

        self.intergated_embeddings = self.attention_transform(support_embedding, embedding)
        print(self.integrated_embeddings)
        return self.classifier(self.intergated_embeddings).squeeze(-1).squeeze(-1)

