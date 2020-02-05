import torch


# Refactored to pytorch
# https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device='cpu'):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, 0)
        batch_var = torch.var(x, 0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class MeanStdNormalizer():
    def __init__(self, device='cpu'):
        self.rms = None
        self.clip = 10.0
        self.epsilon = 1e-8
        self.device = device

    def __call__(self, x):
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:], device=self.device)
        self.rms.update(x)
        return torch.clamp((x - self.rms.mean) / torch.sqrt(self.rms.var + self.epsilon),
                           min=-self.clip, max=self.clip)
