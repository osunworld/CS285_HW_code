import torch
from torch import nn

import cs285.infrastructure.pytorch_util as ptu

class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        # (s,a) -> scalar Q(s,a) 근사 MLP
        # 입력은 상태벡터와 행동벡터를 이어붙인 길이(ob_dim + ac_dim)
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)
    
    def forward(self, obs, acs):
        # obs: (batch, ob_dim), acs: (batch, ac_dim)
        # cat 후 shape: (batch, ob_dim + ac_dim)
        # 출력은 (batch, 1) -> squeeze(-1)로 (batch,)로 맞춘다.
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)
