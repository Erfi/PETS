import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from .DotmapUtils import get_required_argument
from .config.utils import swish, get_affine_params

TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DynModel(nn.Module):
    def __init__(self, ensemble_size, h_units, in_features, out_features):
        super().__init__()

        self.num_nets = ensemble_size

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, h_units)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, h_units, h_units)
        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, h_units, h_units)

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, h_units, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros(1, in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):

        lin0_decays = 0.0001 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, : self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2 :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)


def nn_constructor(model_init_cfg):

    ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")
    input_dim = get_required_argument(model_init_cfg, "input_dim", "Must provide input dimension to the dynamics model")
    output_dim = get_required_argument(
        model_init_cfg, "output_dim", "Must provide output dimension of the dynamics model"
    )
    h_units = get_required_argument(model_init_cfg, "h_units", "Must provide hidden layer units")

    load_model = model_init_cfg.get("load_model", False)

    assert load_model is False, "Has yet to support loading model"

    model = DynModel(ensemble_size, h_units, input_dim, output_dim * 2).to(TORCH_DEVICE)
    # * 2 because we output both the mean and the variance

    model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

    return model
