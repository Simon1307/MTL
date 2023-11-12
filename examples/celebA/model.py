import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, tasks, weighting):
        super(ResNet, self).__init__()
        self.tasks = tasks
        self.task_num = len(self.tasks)
        if weighting == "UW" or weighting == "UW_CAGrad" or weighting == "UW_PCGrad":
            self.loss_scale = nn.Parameter(torch.tensor([-0.5] * self.task_num), requires_grad=True)
        else:
            self.loss_scale = None

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.head_modules = []
        for i in range(self.task_num):
            self.head_modules.append(nn.Linear(2048, 2))
        self.heads = nn.Sequential(*self.head_modules)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def shared_modules(self):
        return [
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        preds = {}
        for i in range(self.task_num):
            out_i = self.heads[i](out)
            out_i = F.log_softmax(out_i, dim=1)
            preds[self.tasks[i]] = out_i

        return preds, self.loss_scale


def wrap_with_dc_uncertainty(model: nn.Module):
    class UWModel(model):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.gaussian_mean = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.gaussian_std = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.gaussian = torch.distributions.Normal(self.gaussian_mean, self.gaussian_std)
            self.uncertainty_model = nn.Sequential(
                    nn.Linear(1,16),
                    nn.ReLU(),
                    nn.Linear(16,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128, len(self.tasks)))

        def compute_uncertainty(self):
            noise = self.gaussian.sample(sample_shape=(1, 1))
            loss_scale = self.uncertainty_model(noise) / 1. - 0.5
            return loss_scale.squeeze(), noise

        def forward(self, x):
            out, _ = super().forward(x)
            loss_scale, _ = self.compute_uncertainty()
            return out, loss_scale
    return UWModel


def wrap_with_c_uncertainty(model: nn.Module):
    class UWModel(model):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.constant = torch.tensor(1.0, requires_grad=False).cuda()
            self.uncertainty_model = nn.Sequential(
                    nn.Linear(1,16),
                    nn.ReLU(),
                    nn.Linear(16,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128, len(self.tasks)))

        def compute_uncertainty(self):
            loss_scale = self.uncertainty_model(self.constant.unsqueeze(dim=0)) / 1. - 0.5
            return loss_scale.squeeze(), self.constant

        def forward(self, x):
            out, _ = super().forward(x)
            loss_scale, _ = self.compute_uncertainty()
            return out, loss_scale
    return UWModel


def wrap_with_d_uncertainty(model: nn.Module):
    class UWModel(model):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.gaussian_mean = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # move to gpu with model
            self.gaussian_std = torch.nn.Parameter(torch.ones(1), requires_grad=False)  # move to gpu with model
            self.gaussians = []  # 1 distinct input distribution per uncertainty network
            self.uncertainty_model1 = self.create_uncertainty_model()
            self.uncertainty_model2 = self.create_uncertainty_model()
            self.uncertainty_model3 = self.create_uncertainty_model()
            self.uncertainty_model4 = self.create_uncertainty_model()
            self.uncertainty_model5 = self.create_uncertainty_model()
            self.uncertainty_model6 = self.create_uncertainty_model()
            self.uncertainty_model7 = self.create_uncertainty_model()
            self.uncertainty_model8 = self.create_uncertainty_model()
            self.uncertainty_model9 = self.create_uncertainty_model()
            self.uncertainty_model10 = self.create_uncertainty_model()
            self.uncertainty_model11 = self.create_uncertainty_model()
            self.uncertainty_model12 = self.create_uncertainty_model()
            self.uncertainty_model13 = self.create_uncertainty_model()
            self.uncertainty_model14 = self.create_uncertainty_model()
            self.uncertainty_model15 = self.create_uncertainty_model()
            self.uncertainty_model16 = self.create_uncertainty_model()
            self.uncertainty_model17 = self.create_uncertainty_model()
            self.uncertainty_model18 = self.create_uncertainty_model()
            self.uncertainty_model19 = self.create_uncertainty_model()
            self.uncertainty_model20 = self.create_uncertainty_model()
            self.uncertainty_model21 = self.create_uncertainty_model()
            self.uncertainty_model22 = self.create_uncertainty_model()
            self.uncertainty_model23 = self.create_uncertainty_model()
            self.uncertainty_model24 = self.create_uncertainty_model()
            self.uncertainty_model25 = self.create_uncertainty_model()
            self.uncertainty_model26 = self.create_uncertainty_model()
            self.uncertainty_model27 = self.create_uncertainty_model()
            self.uncertainty_model28 = self.create_uncertainty_model()
            self.uncertainty_model29 = self.create_uncertainty_model()
            self.uncertainty_model30 = self.create_uncertainty_model()
            self.uncertainty_model31 = self.create_uncertainty_model()
            self.uncertainty_model32 = self.create_uncertainty_model()
            self.uncertainty_model33 = self.create_uncertainty_model()
            self.uncertainty_model34 = self.create_uncertainty_model()
            self.uncertainty_model35 = self.create_uncertainty_model()
            self.uncertainty_model36 = self.create_uncertainty_model()
            self.uncertainty_model37 = self.create_uncertainty_model()
            self.uncertainty_model38 = self.create_uncertainty_model()
            self.uncertainty_model39 = self.create_uncertainty_model()
            self.uncertainty_model40 = self.create_uncertainty_model()

            self.uncertainty_models = [self.uncertainty_model1,
                                       self.uncertainty_model2,
                                       self.uncertainty_model3,
                                       self.uncertainty_model4,
                                       self.uncertainty_model5,
                                       self.uncertainty_model6,
                                       self.uncertainty_model7,
                                       self.uncertainty_model8,
                                       self.uncertainty_model9,
                                       self.uncertainty_model10,
                                       self.uncertainty_model11,
                                       self.uncertainty_model12,
                                       self.uncertainty_model13,
                                       self.uncertainty_model14,
                                       self.uncertainty_model15,
                                       self.uncertainty_model16,
                                       self.uncertainty_model17,
                                       self.uncertainty_model18,
                                       self.uncertainty_model19,
                                       self.uncertainty_model20,
                                       self.uncertainty_model21,
                                       self.uncertainty_model22,
                                       self.uncertainty_model23,
                                       self.uncertainty_model24,
                                       self.uncertainty_model25,
                                       self.uncertainty_model26,
                                       self.uncertainty_model27,
                                       self.uncertainty_model28,
                                       self.uncertainty_model29,
                                       self.uncertainty_model30,
                                       self.uncertainty_model31,
                                       self.uncertainty_model32,
                                       self.uncertainty_model33,
                                       self.uncertainty_model34,
                                       self.uncertainty_model35,
                                       self.uncertainty_model36,
                                       self.uncertainty_model37,
                                       self.uncertainty_model38,
                                       self.uncertainty_model39,
                                       self.uncertainty_model40,
                                       ]

            for i in range(len(self.tasks)):
                self.gaussians.append(torch.distributions.Normal(self.gaussian_mean, self.gaussian_std))

        def compute_uncertainty(self):
            loss_scales = []
            for i in range(len(self.tasks)):
                noise = self.gaussians[i].sample(sample_shape=(1, 1))
                loss_scale = self.uncertainty_models[i](noise) / 1. - 0.5
                loss_scales.append(loss_scale.squeeze())
            loss_scale = torch.hstack(loss_scales)
            return loss_scale.squeeze(), noise

        def forward(self, x):
            out, _ = super().forward(x)
            loss_scale, _ = self.compute_uncertainty()
            return out, loss_scale

        def create_uncertainty_model(self):
            return nn.Sequential(
                    nn.Linear(1,16),
                    nn.ReLU(),
                    nn.Linear(16,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128,1))

    return UWModel


def wrap_with_uw_mlp_uncertainty(model: nn.Module):
    class UWModel(model):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.constant = torch.tensor(1.0, requires_grad=False).cuda()
            self.uncertainty_model1 = self.create_uncertainty_model()
            self.uncertainty_model2 = self.create_uncertainty_model()
            self.uncertainty_model3 = self.create_uncertainty_model()
            self.uncertainty_model4 = self.create_uncertainty_model()
            self.uncertainty_model5 = self.create_uncertainty_model()
            self.uncertainty_model6 = self.create_uncertainty_model()
            self.uncertainty_model7 = self.create_uncertainty_model()
            self.uncertainty_model8 = self.create_uncertainty_model()
            self.uncertainty_model9 = self.create_uncertainty_model()
            self.uncertainty_model10 = self.create_uncertainty_model()
            self.uncertainty_model11 = self.create_uncertainty_model()
            self.uncertainty_model12 = self.create_uncertainty_model()
            self.uncertainty_model13 = self.create_uncertainty_model()
            self.uncertainty_model14 = self.create_uncertainty_model()
            self.uncertainty_model15 = self.create_uncertainty_model()
            self.uncertainty_model16 = self.create_uncertainty_model()
            self.uncertainty_model17 = self.create_uncertainty_model()
            self.uncertainty_model18 = self.create_uncertainty_model()
            self.uncertainty_model19 = self.create_uncertainty_model()
            self.uncertainty_model20 = self.create_uncertainty_model()
            self.uncertainty_model21 = self.create_uncertainty_model()
            self.uncertainty_model22 = self.create_uncertainty_model()
            self.uncertainty_model23 = self.create_uncertainty_model()
            self.uncertainty_model24 = self.create_uncertainty_model()
            self.uncertainty_model25 = self.create_uncertainty_model()
            self.uncertainty_model26 = self.create_uncertainty_model()
            self.uncertainty_model27 = self.create_uncertainty_model()
            self.uncertainty_model28 = self.create_uncertainty_model()
            self.uncertainty_model29 = self.create_uncertainty_model()
            self.uncertainty_model30 = self.create_uncertainty_model()
            self.uncertainty_model31 = self.create_uncertainty_model()
            self.uncertainty_model32 = self.create_uncertainty_model()
            self.uncertainty_model33 = self.create_uncertainty_model()
            self.uncertainty_model34 = self.create_uncertainty_model()
            self.uncertainty_model35 = self.create_uncertainty_model()
            self.uncertainty_model36 = self.create_uncertainty_model()
            self.uncertainty_model37 = self.create_uncertainty_model()
            self.uncertainty_model38 = self.create_uncertainty_model()
            self.uncertainty_model39 = self.create_uncertainty_model()
            self.uncertainty_model40 = self.create_uncertainty_model()

            self.uncertainty_models = [self.uncertainty_model1,
                                       self.uncertainty_model2,
                                       self.uncertainty_model3,
                                       self.uncertainty_model4,
                                       self.uncertainty_model5,
                                       self.uncertainty_model6,
                                       self.uncertainty_model7,
                                       self.uncertainty_model8,
                                       self.uncertainty_model9,
                                       self.uncertainty_model10,
                                       self.uncertainty_model11,
                                       self.uncertainty_model12,
                                       self.uncertainty_model13,
                                       self.uncertainty_model14,
                                       self.uncertainty_model15,
                                       self.uncertainty_model16,
                                       self.uncertainty_model17,
                                       self.uncertainty_model18,
                                       self.uncertainty_model19,
                                       self.uncertainty_model20,
                                       self.uncertainty_model21,
                                       self.uncertainty_model22,
                                       self.uncertainty_model23,
                                       self.uncertainty_model24,
                                       self.uncertainty_model25,
                                       self.uncertainty_model26,
                                       self.uncertainty_model27,
                                       self.uncertainty_model28,
                                       self.uncertainty_model29,
                                       self.uncertainty_model30,
                                       self.uncertainty_model31,
                                       self.uncertainty_model32,
                                       self.uncertainty_model33,
                                       self.uncertainty_model34,
                                       self.uncertainty_model35,
                                       self.uncertainty_model36,
                                       self.uncertainty_model37,
                                       self.uncertainty_model38,
                                       self.uncertainty_model39,
                                       self.uncertainty_model40,
                                       ]

        def compute_uncertainty(self):
            loss_scales = []
            for i in range(len(self.tasks)):
                loss_scale = self.uncertainty_models[i](self.constant.unsqueeze(dim=0)) / 1. - 0.5
                loss_scales.append(loss_scale.squeeze())
            loss_scale = torch.hstack(loss_scales)
            return loss_scale.squeeze(), self.constant

        def forward(self, x):
            out, _ = super().forward(x)
            loss_scale, _ = self.compute_uncertainty()
            return out, loss_scale

        def create_uncertainty_model(self):
            return nn.Sequential(
                    nn.Linear(1,16),
                    nn.ReLU(),
                    nn.Linear(16,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128,1))

    return UWModel
