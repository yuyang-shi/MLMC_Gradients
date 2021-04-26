from torch.distributions import Normal, Bernoulli, Independent


def independent_bernoulli(logits):
    return Independent(Bernoulli(logits=logits), reinterpreted_batch_ndims=1)


def independent_normal(mu, sig):
    return Independent(Normal(mu, sig), reinterpreted_batch_ndims=1)