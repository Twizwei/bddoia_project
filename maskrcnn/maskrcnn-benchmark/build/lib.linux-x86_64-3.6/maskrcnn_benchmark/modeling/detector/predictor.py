import torch
import torch.nn as nn

from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, config, select_num=10, class_num=4, side=False, is_cat=False):
        super(Predictor, self).__init__()
        self.selector = Selector()
        self.select_num = select_num
        self.sftmax = nn.Softmax(dim=0)
        self.side = side
        self.is_cat = is_cat

        # ResNet Head
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )
        self.head.load_state_dict(torch.load('/data6/SRIP19_SelfDriving/Outputs/layer4.pth')) #TODO: how to avoid this? Maybe we can involve the weights into model_final.pth


        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=14)
        if self.is_cat:
            self.avg1 = nn.AdaptiveAvgPool3d(output_size=1)
            self.avg2 = nn.AdaptiveAvgPool2d(output_size=1)
            self.fc1 = nn.Linear(4096, 100)
            self.relu1 = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(100, class_num)
            self.drop = nn.Dropout(p=0.25)
            if self.side:
                self.fc_side1 = nn.Linear(4096, 100)
                self.relu_side1 = nn.ReLU(inplace=True)
                self.fc_side2 = nn.Linear(100, 21)
        else:
            self.avg = nn.AdaptiveAvgPool2d(output_size=1)
            self.fc2 = nn.Linear(2048, class_num)
            self.drop = nn.Dropout(p=0.25)
            if self.side:
                self.fc_side = nn.Linear(2048, 21)


    def forward(self, x):
        scores = self.selector(x['roi_features'])
        scores, idx = torch.sort(scores, dim=0, descending=True)
        scores_logits = self.sftmax(scores)
        # print(idx.shape[0])
        
        if self.select_num < idx.shape[0]: # in case that too few objects detected
            idx = idx[:self.select_num].reshape(self.select_num)
            select_features = x['roi_features'][idx]
            select_features *= scores_logits[:self.select_num] # shape(select_num, 1024, 14, 14)
        else:
            idx = idx.reshape(idx.shape[0])
            select_features = x['roi_features'][idx]
            select_features *= scores_logits[:idx.shape[0]]

        if self.is_cat:
            select_features = self.head(select_features) # (select_num, 2048, 7, 7)
            select_features = select_features.transpose(1,0) # (2048, select_num, 7, 7)
            select_features = self.avg1(select_features) # (2048, 1, 1, 1)
            select_features = select_features.view(select_features.size(1), -1) # (1, 2048)

            glob_feature = self.avgpool_glob(x['glob_feature']) # shape(1, 1024, 14, 14)
            glob_feature = self.avg2(self.head(glob_feature)) # (1, 2048, 1, 1)
            glob_feature = glob_feature.view(glob_feature.size()[0], -1)

            tmp = torch.cat((select_features, glob_feature), dim=1) # (1, 4096)
            del select_features, glob_feature, scores, idx, scores_logits
            x = self.drop(self.relu1(self.fc1(tmp))) # (1, 64)
            x = self.drop(self.fc2(x)) # (1, num_class)

            if self.side:
                side = self.drop(self.relu_side1(self.fc_side1(tmp)))
                side = self.drop(self.fc_side2(side))
        else: 
            select_feature = torch.sum(select_features, dim=0).unsqueeze(0) # (1, 1024, 14, 14)
            glob_feature = self.avgpool_glob(x['glob_feature']) # shape(1, 1024, 14, 14)
            del x
            x = self.head(glob_feature + select_feature) # shape(1, 2048, 7, 7)
            del glob_feature, select_feature, scores, idx, scores_logits

            x = self.avg(x) # shape(1, 2048, 1, 1)
            tmp = x.view(x.size(0), -1) # shape(1, 2048)
            x = self.fc2(tmp) # shape(1, num_class)
            # x = self.drop(x)
            # print(self.side)
            if self.side:
                side = (self.fc_side(tmp))

        return (x, side) if self.side else x






class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.conv = nn.Conv2d(1024, 1, 14)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x.shape (N, 1024, 14, 14)
        weights = self.conv(x) # weights.shape (N, 1, 1, 1)
        weights = self.relu(weights)

        return weights

def build_predictor(cfg, side, is_cat):
    model = Predictor(cfg, side=side, is_cat=is_cat)
    return model
