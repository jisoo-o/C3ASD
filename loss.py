import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(128, 2)
		
	def forward(self, x, labels = None, r = 1):	
		x = x.squeeze(1)
		x = self.FC(x)
		if labels == None:
			predScore = x[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			x1 = x / r
			x1 = F.softmax(x1, dim = -1)[:,1]
			nloss = self.criterion(x1, labels.float())
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum


class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels, r = 1):	
		x = x.squeeze(1)
		x = self.FC(x)
		
		x = x / r
		x = F.softmax(x, dim = -1)

		nloss = self.criterion(x[:,1], labels.float())
		return nloss


class lossA(nn.Module):
	"""Audio-only loss for prediction-level consistency"""
	def __init__(self):
		super(lossA, self).__init__()
		self.FC = nn.Linear(128, 2)
	
	def forward(self, x):
		x = x.squeeze(1)
		x = self.FC(x)
		x = F.softmax(x, dim=-1)
		return x


class InterModalityConsistency(nn.Module):
	"""
	Inter-modality consistency loss using cosine similarity (speaking frames only).
	"""
	def __init__(self):
		super(InterModalityConsistency, self).__init__()
		self.cosine_similarity = nn.CosineSimilarity(dim=-1)

	def forward(self, audio_embed, visual_embed, labels):
		mask = labels == 1
		if not mask.any():
			return torch.tensor(0.0, device=audio_embed.device)
		similarity = self.cosine_similarity(audio_embed[mask], visual_embed[mask])
		loss = (1 - similarity).mean()
		return loss


class IntraModalityConsistency(nn.Module):
	"""
	Intra-modality consistency loss using supervised contrastive objective.
	"""
	def __init__(self, temperature=0.07):
		super(IntraModalityConsistency, self).__init__()
		self.temperature = temperature
	
	def forward(self, embeddings, labels, group_ids=None):
		embeddings = F.normalize(embeddings, p=2, dim=1)
		logits = torch.matmul(embeddings, embeddings.t()) / self.temperature
		n = logits.size(0)

		labels = labels.view(-1, 1)
		mask_pos = (labels == labels.t())
		logits_mask = ~torch.eye(n, dtype=torch.bool, device=logits.device)
		if group_ids is not None:
			group_ids = group_ids.view(-1, 1)
			mask_group = (group_ids == group_ids.t())
			mask_pos = mask_pos & mask_group
			logits_mask = logits_mask & mask_group

		mask_pos = mask_pos & logits_mask

		logits_max, _ = torch.max(logits, dim=1, keepdim=True)
		logits = logits - logits_max.detach()
		exp_logits = torch.exp(logits) * logits_mask.float()
		log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

		pos_count = mask_pos.sum(dim=1)
		valid_mask = pos_count > 0
		if valid_mask.any():
			mean_log_prob_pos = (mask_pos.float() * log_prob).sum(dim=1) / pos_count.float().clamp_min(1.0)
			loss = -mean_log_prob_pos[valid_mask].mean()
		else:
			loss = torch.tensor(0.0, device=embeddings.device)

		return loss


class PredictionLevelConsistency(nn.Module):
	"""
	Prediction-level consistency with confidence-masked probability MSE.
	"""
	def __init__(self):
		super(PredictionLevelConsistency, self).__init__()
		self.conf_threshold = 0.7
	
	def forward(self, p_av, p_a, p_v):
		p_teacher = p_av.detach()
		conf = p_teacher.max(dim=1).values
		mask = conf >= self.conf_threshold
		if not mask.any():
			return torch.tensor(0.0, device=p_av.device)

		t = p_teacher[mask, 1]
		a = p_a[mask, 1]
		v = p_v[mask, 1]
		loss_a = ((a - t) ** 2).mean()
		loss_v = ((v - t) ** 2).mean()
		return loss_a + loss_v
