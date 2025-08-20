from numpy.random import randint
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from .BalancedBatchSampler import BalancedBatchSampler
from models.Supportnet import Supportnet


def get_balanced_loader(dataset, n_classes, n_samples):
    labels = [example[1] for example in dataset]  # Assuming dataset[i] = (X, y)
    sampler = BalancedBatchSampler(labels, n_classes=n_classes, n_samples=n_samples)
    return DataLoader(dataset, batch_sampler=sampler)


'''
Create support/query sets from a batch.
'''
def sample_episode(X, y, num_classes, n_prototype, n_query):
    """
    Create support/query sets from a batch.

    Inputs:
        X: [batch_size, channels, time]
        y: [batch_size] (LongTensor of class labels)
        num_classes: number of classes per episode
        n_prototype: number of support examples per class
        n_query: number of query examples per class

    Returns:
        prototype_x, prototype_y, query_x, query_y
    """
    device = X.device
    unique_classes = torch.unique(y)
    assert len(unique_classes) >= num_classes, "Not enough classes in batch"

    selected_classes = unique_classes[torch.randperm(len(unique_classes))[:num_classes]]

    prototype_x = []
    prototype_y = []
    query_x = []
    query_y = []

    for cls in selected_classes:
        class_mask = (y == cls)
        cls_indices = class_mask.nonzero(as_tuple=True)[0]
        cls_indices = cls_indices[torch.randperm(len(cls_indices))]

        assert len(cls_indices) >= n_prototype + n_query, f"Not enough examples for class {cls.item()}"

        prototype_indices = cls_indices[:n_prototype]
        query_indices = cls_indices[n_prototype:n_prototype + n_query]

        prototype_x.append(X[prototype_indices])
        prototype_y.append(y[prototype_indices])
        query_x.append(X[query_indices])
        query_y.append(y[query_indices])

    prototype_x = torch.cat(prototype_x, dim=0).to(device)
    prototype_y = torch.cat(prototype_y, dim=0).to(device)
    query_x = torch.cat(query_x, dim=0).to(device)
    query_y = torch.cat(query_y, dim=0).to(device)

    return prototype_x, prototype_y, query_x, query_y


"""
Episodic training: construct one support/query episode from each batch.
Assumes support_encoder, task_encoder, and attention_transform_with_prototypes
are defined in the model.
"""
def train_one_epoch_episodic(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn, 
    optimizer,
    scheduler: LRScheduler, 
    epoch: int, 
    device="cuda", 
    num_classes = 4,
    print_batch_stats=False
):

    # Episodic setup
    batch_size = dataloader.batch_size
    n_support = batch_size // (2 * num_classes)
    n_query = batch_size // (2 * num_classes)

    # Set the model to training mode
    model.train()  
    train_loss, correct = 0, 0

    progress_bar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader),
        disable=not print_batch_stats
    )

    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # ===== Sample support/query set from batch =====
        try:
            support_x, support_y, query_x, query_y = sample_episode(
                X, y, num_classes=num_classes,
                n_support=n_support, n_query=n_query
            )
        except AssertionError as e:
            # Skip this batch if it doesn't contain enough data per class
            if print_batch_stats:
                print(f"Skipping batch {batch_idx} (not enough samples per class)")
            continue
        
        # ===== Encode support and query sets =====
        _ = model.support_encoder(support_x)
        support_emb = model.support_encoder.get_embeddings()
        # remove the singleton dimension
        support_emb = support_emb.squeeze(-1)

        _ = model.encoder(query_x)

        task_emb = model.encoder.get_embeddings()
        # remove the singleton dimension
        task_emb = task_emb.squeeze(-1)

        task_emb_adapted = model.attention_transform_with_prototypes(
            support_emb, support_y, task_emb, num_classes=num_classes
        )

        # ===== Final classification head =====
        # [batch_size_query, num_classes]
        logits = model.classifier(task_emb_adapted).squeeze(-1).squeeze(-1)

        # ===== Loss and optimization =====
        loss = loss_fn(logits, query_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (logits.argmax(1) == query_y).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}"
            )

    # Update the learning rate
    scheduler.step()

    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


@torch.no_grad()
def test_model_episodic(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn, 
    n_proto: int,
    n_query: int,
    num_classes=4,
    max_test_episodes=30,
    print_batch_stats=False, 
    device="cuda"
):
    model.eval()
    test_loss, correct, total_query = 0.0, 0.0, 0

    if print_batch_stats:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)

    for batch_idx, (X, y, _) in progress_bar:

        if batch_idx > max_test_episodes:
            break

        X, y = X.to(device), y.to(device)

        try:
            proto_x, proto_y, query_x, query_y = sample_episode(
                X, y, num_classes=num_classes,
                n_prototype=n_proto, n_query=n_query
            )
        except AssertionError:
            if print_batch_stats:
                print(
                    f'Skipping batch {batch_idx} in test' 
                    '(not enough examples per class)'
                )
            continue

        # === Embed ===
        with torch.no_grad():
            # ===== Encode prototype and query samples =====
            _ = model.support_encoder(proto_x)
            proto_support_emb = model.support_encoder.get_embeddings()
            proto_support_emb = proto_support_emb.squeeze(-1)
            _ = model.encoder(proto_x)
            proto_task_emb = model.encoder.get_embeddings()
            proto_task_emb = proto_task_emb.squeeze(-1)

            _ = model.support_encoder(query_x)
            query_support_emb = model.support_encoder.get_embeddings()
            query_support_emb = query_support_emb.squeeze(-1)
            _ = model.encoder(query_x)
            query_task_emb = model.encoder.get_embeddings()
            query_task_emb = query_task_emb.squeeze(-1)

            # ===== Attention transformation =====
            query_task_emb_adapted = model.attention_transform_with_prototypes(
                proto_support_emb, proto_task_emb, proto_y, 
                query_support_emb, query_task_emb,
                num_classes=num_classes
            )

            logits = model.classifier(query_task_emb_adapted).squeeze(-1).squeeze(-1)

            loss = loss_fn(logits, query_y).item()
            preds = logits.argmax(dim=1)

        test_loss += loss
        correct += (preds == query_y).sum().item()
        total_query += query_y.size(0)

        if print_batch_stats:
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss:.6f}"
            )

    avg_loss = test_loss / len(dataloader)
    accuracy = correct / total_query if total_query > 0 else 0.0

    print(f"Test Accuracy: {100 * accuracy:.1f}%, Test Loss: {avg_loss:.6f}\n")
    return avg_loss, accuracy


"""
Meta-learning training loop where each subject is treated as a separate task.
In each iteration, we sample one subject and construct a support/query split from it.
"""
def train_one_epoch_meta_subject(
    subject_loaders,
    model: Supportnet,
    loss_fn,
    optimizer,
    scheduler: LRScheduler,
    n_proto: int,
    n_query: int,
    device="cuda",
    num_classes=4,
    print_batch_stats=False
):
    
    model.train()
    total_loss, correct, total_query = 0.0, 0.0, 0

    subject_cnt = len(subject_loaders)

    # for batch_idx, subject_loader in enumerate(subject_loaders):
    for _ in range(subject_cnt):
        # Randomize access order
        subject_loader = subject_loaders[randint(0, subject_cnt)]
        X, y, _ = next(iter(subject_loader))
        X, y = X.to(device), y.to(device)

        try:
            proto_x, proto_y, query_x, query_y = sample_episode(
                X, y, num_classes=num_classes, n_prototype=n_proto, n_query=n_query
            )
        except AssertionError:
            if print_batch_stats:
                print(f"Skipping batch {batch_idx} in train (not enough class examples)")
            continue

        # ===== Encode prototype and query samples =====
        _ = model.support_encoder(proto_x)
        proto_support_emb = model.support_encoder.get_embeddings()
        proto_support_emb = proto_support_emb.squeeze(-1)
        _ = model.encoder(proto_x)
        proto_task_emb = model.encoder.get_embeddings()
        proto_task_emb = proto_task_emb.squeeze(-1)

        _ = model.support_encoder(query_x)
        query_support_emb = model.support_encoder.get_embeddings()
        query_support_emb = query_support_emb.squeeze(-1)
        _ = model.encoder(query_x)
        query_task_emb = model.encoder.get_embeddings()
        query_task_emb = query_task_emb.squeeze(-1)

        # ===== Attention transformation =====
        query_task_emb_adapted = model.attention_transform_with_prototypes(
            proto_support_emb, proto_task_emb, proto_y, 
            query_support_emb, query_task_emb,
            num_classes=num_classes
        )

        # ===== Final classification head =====
        # [batch_size_query, num_classes]
        logits = model.classifier(query_task_emb_adapted).squeeze(-1).squeeze(-1)

        # print('Classification done')

        # ===== Loss and optimization =====
        loss = loss_fn(logits, query_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == query_y).sum().item()
        total_query += query_y.size(0)

    scheduler.step()
    avg_loss = total_loss / len(subject_loaders)
    accuracy = correct / total_query if total_query > 0 else 0.0

    return avg_loss, accuracy