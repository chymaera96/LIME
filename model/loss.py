import torch
import torch.nn.functional as F

def weighted_mse_loss(emb_SSM, label_SSM, audio_len):
    batch_size = emb_SSM.shape[0]
    weights = torch.reciprocal(audio_len.float())
    sq_error = (emb_SSM - label_SSM) **2
    sq_error = sq_error.view(batch_size, -1)
    loss = torch.sum(sq_error * weights.view(-1, 1), axis=1)
    return torch.mean(loss)

def assymetric_loss(emb_SSM, label_SSM, audio_len):
    batch_size = emb_SSM.shape[0]
    weights = torch.reciprocal(torch.Tensor(audio_len).float())
    error = F.relu(label_SSM - emb_SSM)
    error = error.view(batch_size, -1)
    loss = torch.sum(error * weights.view(-1, 1), axis=1)
    return torch.mean(loss)