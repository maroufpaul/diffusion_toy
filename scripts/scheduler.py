# scheduler.py
import torch


#x0: original clean data [B,C,D,H,W] batchsize, channeles, depth height width.
#t time step for each shape [B]
def q_sample(x0: torch.Tensor, t: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """
    x0: (B, C, D, H, W)
    t:  (B,) integer timesteps
    """
    noise = torch.randn_like(x0)# generate ramdom gaussin noise same shape to x0
    # compute alpha_t = (1-beta)^t as a (B,1,1,1,1) tensor for broadcasting
    alpha_t = (1 - beta) ** t.float()# t = 5 → alpha_t = (0.9)^5 ≈ 0.59 → 59% signal, 41% noise
    # reshape to (B,1,1,1,1)
    alpha_t = alpha_t.view(-1, 1, 1, 1, 1) #[4] --> [4,1,1,1,1]
    return alpha_t.sqrt() * x0 + (1 - alpha_t).sqrt() * noise # CLEAN PART + RANDOM NOISE

#alpha_t is a value between 0 and 1
# At t = 0, it's 1 → all clean
#At t = 10, it's small (like 0.3) → mostly noise

