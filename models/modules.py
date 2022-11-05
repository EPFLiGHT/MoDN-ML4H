# Neural netwrok modules : Definition of the encoder, decoder and state modules of the MoDN
# MLP Neural Network baseline

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpoctEncoder(nn.Module):
    """
    Feature encoder
    """

    def __init__(self, STATE_SIZE, hidden_size=32):
        super(EpoctEncoder, self).__init__()
        self.fc1 = nn.Linear(1 + STATE_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, STATE_SIZE)

    def forward(self, state, x):
        x = F.relu(self.fc1(torch.cat([x, state], axis=-1)))
        return state + self.fc2(x)


class EpoctBinaryDecoder(nn.Module):
    """
    Disease decoder
    """

    def __init__(self, STATE_SIZE, hidden_size=10):
        super(EpoctBinaryDecoder, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 2, bias=True)

    def forward(self, x):
        return F.log_softmax(self.fc1(x), dim=1)


class InitState(nn.Module):
    """Initial state module"""

    def __init__(self, STATE_SIZE):
        super(InitState, self).__init__()
        self.STATE_SIZE = STATE_SIZE
        self.state_value = torch.nn.Parameter(
            torch.randn([1, STATE_SIZE], requires_grad=True)
        )

    def forward(self, n_data_points):
        init_tensor = torch.tile(self.state_value, [n_data_points, 1])
        return init_tensor


class BaselineBinary2layersMLP(nn.Module):
    """MLP baseline"""

    def __init__(self, input_size, hidden_size=10):
        super(BaselineBinary2layersMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc3(F.relu(x)), dim=1)


class EpoctCategoricalDecoder(nn.Module):
    """Categorical feature decoder"""

    def __init__(self, STATE_SIZE, num_categories, hidden_size=10):
        super(EpoctCategoricalDecoder, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_categories, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class EpoctDistributionDecoder(nn.Module):
    """Feature distribution decoder approximating mean and variance
    of distribution"""

    def __init__(self, STATE_SIZE, hidden_size=10):
        super(EpoctDistributionDecoder, self).__init__()
        self.fc1_mu = nn.Linear(STATE_SIZE, hidden_size, bias=True)
        self.fc2_mu = nn.Linear(hidden_size, 1, bias=True)
        self.fc1_sigma = nn.Linear(STATE_SIZE, hidden_size, bias=True)
        self.fc2_sigma = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        mu = self.fc2_mu(F.relu(self.fc1_mu(x)))
        log_sigma = self.fc2_sigma(F.relu(self.fc1_sigma(x)))
        return mu, log_sigma


class BaselineBinary2layersMLP(nn.Module):
    """MLP baseline"""

    def __init__(self, input_size, hidden_size=10):
        super(BaselineBinary2layersMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc3(F.relu(x)), dim=1)
