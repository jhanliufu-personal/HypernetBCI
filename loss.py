import torch
import torch.nn.functional as F

'''
NT-Xent loss, adapted from SimCLR
'''
def nt_xent_loss(embeddings, labels, temperature=0.5):
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Mask to remove self-similarity
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    # Calculate contrastive loss
    exp_sim = torch.exp(similarity_matrix)
    sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
    log_prob = similarity_matrix - torch.log(sum_exp_sim)
    
    # Create a mask for positive pairs
    labels = labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float()
    
    # Calculate loss for positive pairs
    loss = -(positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1)
    loss = loss.mean()
    
    return loss


class contrastive_loss_btw_subject(torch.nn.Module):

    def __init__(self, subject_cnt, emb_cnt_per_subj, batch_size, temperature=0.5, device='cuda'):
        super(contrastive_loss_btw_subject, self).__init__()
        assert subject_cnt * emb_cnt_per_subj == batch_size, 'Batch size does not match'
        self.subject_cnt = subject_cnt
        self.emb_cnt_per_subj = emb_cnt_per_subj
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._dot_simililarity

        self.positive_mask = self._create_positive_mask()
        self.negative_mask = self._create_negative_mask()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _create_positive_mask(self):
        n = self.subject_cnt
        k = self.emb_cnt_per_subj
        mask = torch.zeros(n * k, n * k, dtype=torch.bool)
        
        for i in range(n):
            start = i * k
            end = start + k
            # Create submask for k x k block
            submask = torch.zeros(k, k, dtype=torch.bool)
            # Set the first row all True except the diagonal
            submask[0, :] = True
            submask[0, 0] = False
            # Set one other non-diagonal entry to True
            submask[1, 1] = True
            # Add to overall mask
            mask[start:end, start:end] = submask
        
        return mask

    def _create_negative_mask(self):
        n = self.subject_cnt
        k = self.emb_cnt_per_subj
        mask = torch.ones(n * k, n * k, dtype=torch.bool)
        for i in range(n):
            start = i * k
            end = start + k
            # Zero out diagonal blocks
            mask[start:end, start:end] = 0
        return mask
    
    @staticmethod
    def _dot_simililarity(x, y):
        return torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
    
    def forward(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = self.similarity_function(embeddings, embeddings)

        positives = similarity_matrix[self.positive_mask].view(self.batch_size, 1)
        negatives = similarity_matrix[self.negative_mask].view(self.batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(self.batch_size).long()
        loss = self.criterion(logits, labels)

        return loss / self.batch_size
