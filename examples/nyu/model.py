import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
from collections import OrderedDict


class SegNet_MTAN(nn.Module):
    def __init__(self, tasks, weighting):
        super(SegNet_MTAN, self,).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.tasks = tasks
        if weighting == "UW" or weighting == "UW_CAGrad" or weighting == "UW_PCGrad":
            task_num = len(self.tasks)
            self.loss_scale = nn.Parameter(torch.tensor([-0.5] * task_num), requires_grad=True)
        else:
            self.loss_scale = None

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(
                    nn.Sequential(
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                    )
                )
                self.conv_block_dec.append(
                    nn.Sequential(
                        self.conv_layer([filter[i], filter[i]]),
                        self.conv_layer([filter[i], filter[i]]),
                    )
                )

        # define task attention layers
        self.encoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]
        )
        self.decoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])]
        )
        self.encoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[1]])]
        )
        self.decoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[0]])]
        )

        for j in range(3):
            if j < 2:
                self.encoder_att.append(
                    nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])
                )
                self.decoder_att.append(
                    nn.ModuleList(
                        [self.att_layer([2 * filter[0], filter[0], filter[0]])]
                    )
                )
            for i in range(4):
                self.encoder_att[j].append(
                    self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]])
                )
                self.decoder_att[j].append(
                    self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]])
                )

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 2]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i]])
                )
            else:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            self.encoder_block_att,
            self.decoder_block_att,
            self.down_sampling,
            self.up_sampling,
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[0],
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=1,
                    padding=0,
                ),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channel[0],
                out_channels=channel[1],
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channel[1],
                out_channels=channel[2],
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1)
                    )
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_encoder[i][-1][-1],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_decoder[i][j - 1][2],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        #t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t1_pred = self.pred_task1(atten_decoder[0][-1][-1])  # no softmax
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        preds = {self.tasks[0]: t1_pred,
                 self.tasks[1]: t2_pred,
                 self.tasks[2]: t3_pred}

        return preds, self.loss_scale


class SegNet(nn.Module):
    def __init__(self, tasks, weighting):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.tasks = tasks
        if weighting == "UW" or weighting == "UW_CAGrad" or weighting == "UW_PCGrad":
            task_num = len(self.tasks)
            self.loss_scale = nn.Parameter(torch.tensor([-0.5] * task_num), requires_grad=True)
        else:
            self.loss_scale = None

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            self.down_sampling,
            self.up_sampling,
        ]

    def params_seg_optim(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            self.down_sampling,
            self.up_sampling,
            self.pred_task1
        ]

    def params_depth_optim(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            self.down_sampling,
            self.up_sampling,
            self.pred_task2
        ]

    def params_normal_optim(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            self.down_sampling,
            self.up_sampling,
            self.pred_task3
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task prediction layers
        #t1_pred = F.log_softmax(self.pred_task1(g_decoder[-1][-1]), dim=1)
        t1_pred = self.pred_task1(g_decoder[-1][-1])  # no softmax
        t2_pred = self.pred_task2(g_decoder[-1][-1])
        t3_pred = self.pred_task3(g_decoder[-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        preds = {self.tasks[0]: t1_pred,
                 self.tasks[1]: t2_pred,
                 self.tasks[2]: t3_pred}

        return preds, self.loss_scale


class DeepLabV3(nn.Module):
    def __init__(self, tasks, weighting, num_classes, pretrained, **kwargs):
        super(DeepLabV3, self).__init__()
        # output stride = 8
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
        resnet_type = kwargs["resnet"]
        if resnet_type == "RESNET101":
            backbone = resnet.resnet101(
                pretrained=pretrained,
                replace_stride_with_dilation=replace_stride_with_dilation)
        elif resnet_type == "RESNET50":
            backbone = resnet.resnet50(
                pretrained=pretrained,
                replace_stride_with_dilation=replace_stride_with_dilation)
        inplanes = 2048
        low_level_planes = 256
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        decoders = nn.ModuleDict(
            {tasks[0]: DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes[0], aspp_dilate),
             tasks[1]: DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes[1], aspp_dilate),
             tasks[2]: DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes[2], aspp_dilate)
             })
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)  # shared layers

        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        if weighting == "UW" or weighting == "UW_CAGrad" or weighting == "UW_PCGrad":
            task_num = len(self.tasks)
            self.loss_scale = nn.Parameter(torch.tensor([-0.5] * task_num), requires_grad=True)
        else:
            self.loss_scale = None

    def shared_modules(self):
        return [
            self.backbone
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        # heads
        t1_pred = self.decoders["segmentation"](features)
        t1_pred = F.interpolate(t1_pred, size=input_shape, mode='bilinear', align_corners=False)

        t2_pred = self.decoders["depth"](features)
        t2_pred = F.interpolate(t2_pred, size=input_shape, mode='bilinear', align_corners=False)

        t3_pred = self.decoders["normal"](features)
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        t3_pred = F.interpolate(t3_pred, size=input_shape, mode='bilinear', align_corners=False)

        preds = {self.tasks[0]: t1_pred,
                 self.tasks[1]: t2_pred,
                 self.tasks[2]: t3_pred}

        return preds, self.loss_scale


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'):  # if using hrnet, you need to take care of transition
                if name == 'transition1':  # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else:  # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else:  # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag:  # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


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
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
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
            self.uncertainty_model1 = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1)).cuda()

            self.uncertainty_model2 = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1)).cuda()

            self.uncertainty_model3 = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1)).cuda()

            self.uncertainty_models = [self.uncertainty_model1, self.uncertainty_model2, self.uncertainty_model3]
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
    return UWModel


def wrap_with_uw_mlp_uncertainty(model: nn.Module):
    class UWModel(model):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.constant = torch.tensor(1.0, requires_grad=False).cuda()
            self.uncertainty_model1 = nn.Sequential(
                    nn.Linear(1,16),
                    nn.ReLU(),
                    nn.Linear(16,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128,1)).cuda()

            self.uncertainty_model2 = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1)).cuda()

            self.uncertainty_model3 = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1)).cuda()

            self.uncertainty_models = [self.uncertainty_model1, self.uncertainty_model2, self.uncertainty_model3]

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
    return UWModel
