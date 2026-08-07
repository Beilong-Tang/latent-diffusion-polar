import torch

def normalize(x, eps=1e-9, return_norm = False):
    """
    Normalize x to be on the unit sphere

    x: [B, C, H, W]

    """
    x_shape = x.shape
    x_flat = torch.flatten(x, start_dim=1)
    norm = x_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    x_flat = x_flat / norm
    if return_norm:
        return x_flat.view(x_shape), norm.squeeze(-1)
    return x_flat.view(x_shape)


def exp_map(p, v):
    """
    exp map on a unit sphere from of point p and tangent vector v
    p: [B,D] or [B, C, H, D]
    v: [B,D] or [B, C, H, D]
    """
    p_shape = p.shape
    bb = p_shape[0]
    p = p.reshape(bb, -1)
    v = v.reshape(bb, -1)

    v_norm = torch.norm(v, dim=-1, keepdim=True)  # ||v||
    theta = v_norm

    res = torch.cos(theta) * p + torch.sin(theta) * (v / (v_norm + 1e-8))
    res = normalize(res)
    return res.reshape(p_shape)

def proj(p, w):
    p_shape = p.shape
    p = p.reshape(p.size(0), -1)
    w = w.reshape(w.size(0), -1)

    inner = (p * w).sum(dim=-1, keepdim=True)
    res = w - inner * p
    return res.reshape(p_shape)



model_dict={}
def get_model(name):
    return model_dict[name]


def get_mean_and_std_rho(dataloader, field='rho'):
    """
    cal normalize(data) to get r and then get rho by log(r)
    it then estimate the mean and std of log(r)
    """
    assert field in ['rho', 'r']
    res = []
    print(f"getting std and mean for the rho of data, field={field}")
    for d in dataloader:
        _, r = normalize(d, return_norm=True)
        if field == 'rho':
            res.append(torch.log(r))
        elif field == 'r':
            res.append(r)
        else:
            raise Exception(f"field: {field} not supported")
    res = torch.cat(res) # [N]
    mean = torch.mean(res).item()
    std = torch.std(res).item()
    del res
    return lambda x: (x - mean) / std