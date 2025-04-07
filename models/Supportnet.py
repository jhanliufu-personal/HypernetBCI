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
    def __init__(self, support_encoder, encoder, classifier, dim=40) -> None:
        super(Supportnet, self).__init__() 
        self.support_encoder = support_encoder
        self.encoder = encoder
        self.classifier = classifier

        # Query from task embedding (per time step)
        self.query_layer = nn.Linear(40, dim)
        # Key from support embedding
        self.key_layer = nn.Linear(40, dim)
        # Value from task embedding (per time step) 
        self.value_layer = nn.Linear(40, dim)  

        self.integrated_embeddings = None

        self.class_prototypes = None


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
        # Shape: [batch_size, ]
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

        # print(f'updated_task_emb shape = {updated_task_emb.shape}')
        return updated_task_emb


    def attention_transform_with_prototypes(
        self, support_emb, support_y, task_emb, num_classes=4
    ):
        """
        Apply attention over class prototypes at each time step independently.

        Returns:
            updated_task_embedding: [batch_size, 144, 40]
        """
        device = task_emb.device
        T, D = task_emb.shape[1], task_emb.shape[2]  # T=144, D=40

        # === Step 1: Compute class prototypes by averaging over support examples
        class_prototypes = torch.zeros((num_classes, T, D), device=device)
        for c in range(num_classes):
            class_mask = (support_y == c)
            # If there is any sample for this label class
            if class_mask.sum() > 0:
                # [144, 40]
                class_prototypes[c] = support_emb[class_mask].mean(dim=0)  

        # === Step 2: Project task embedding (query), prototype (key/value)

        # # Reshape: [batch_size * T, D]
        # task_q = task_emb.reshape(-1, D)
        # # [144, num_classes, 40]
        # proto_kv = class_prototypes.permute(1, 0, 2)  

        output = []
        # Projection per time step — per time step
        for t in range(T):
            q = self.query_layer(task_emb[:, t]) # [batch_size, dim]         
            k = self.key_layer(class_prototypes[:, t]) # [num_classes, dim]         
            v = self.value_layer(class_prototypes[:, t]) # [num_classes, dim]      

            # [batch_size, num_classes]
            attn_scores = torch.matmul(q, k.T)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # [batch_size, dim]
            attended = torch.matmul(attn_weights, v)

            # Optionally add residual
            updated = attended + task_emb[:, t] # [batch_size, dim]           
            output.append(updated.unsqueeze(1)) # Keep time dim                 

        return torch.cat(output, dim=1)


    def forward(self, x):
        # print(x)
        _ = self.support_encoder(x)
        support_embedding = self.support_encoder.get_embeddings()
        # print(f'support_embedding.shape = {support_embedding.shape}')
        _ = self.encoder(x)
        embedding = self.encoder.get_embeddings()
        # print(f'embedding.shape = {embedding.shape}')

        self.integrated_embeddings = self.concatenate_embeddings(support_embedding, embedding)
        # return self.classifier(concatenated_embedding).squeeze(-1).squeeze(-1)

        # self.integrated_embeddings = self.attention_transform(support_embedding, embedding)
        # print(f'self.integrated_embeddings.shape = {self.integrated_embeddings.shape}')
        return self.classifier(self.integrated_embeddings).squeeze(-1).squeeze(-1)

