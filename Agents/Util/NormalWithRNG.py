from torch.distributions import Normal
import torch

class NormalWithRNG(Normal):
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args)
    
    def sample(self, sample_shape=torch.Size(), rng=None):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape), generator=rng)
        
    def rsample(self, sample_shape=torch.Size(), rng=None):
        shape = self._extended_shape(sample_shape)
        if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
            eps = torch.normal(
                torch.zeros(shape, dtype=self.loc.dtype, device=self.loc.device),
                torch.ones(shape, dtype=self.loc.dtype, device=self.loc.device),
                generator=rng
            )
        else:
            eps = torch.empty(shape, dtype=self.loc.dtype, device=self.loc.device).normal_(generator=rng)
        return self.loc + eps * self.scale