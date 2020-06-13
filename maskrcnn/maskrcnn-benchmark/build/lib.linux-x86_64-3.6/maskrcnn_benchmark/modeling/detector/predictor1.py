import torch
import torch.nn as nn

# from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, cfg=None, select_num=5, class_num=4, side=False):
        super(Predictor, self).__init__()
        self.side = side

        # Layers for global feature
        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=7)

        # Selector
        self.selector = Selector()
        self.select_num = select_num
        self.sftmax = nn.Softmax(dim=0)

        # Last
        self.conv1 = nn.Conv2d(2048+1024, 512, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 128, 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128*select_num, class_num)
        if self.side:
            self.fc_side = nn.Linear(128*select_num, 21)


    def forward(self, x):
        # Processing global feature (1, 1024, 45, 80)
        glob = self.avgpool_glob(x['glob_feature']) # (1, 1024, 7, 7)

        # Select objects (N, 2048, 7, 7)
        obj = x['roi_features']
        scores = self.selector(obj)
        scores, idx = torch.sort(scores, dim=0, descending=True)
        scores_logits = self.sftmax(scores)

        if self.select_num < idx.shape[0]: # in case that too few objects detected
            idx = idx[:self.select_num].reshape(self.select_num)
            obj = obj[idx]
            obj *= scores_logits[:self.select_num] # shape(select_num, 2048, 7, 7)
        else:
            idx = idx.reshape(idx.shape[0])
            obj = obj[idx]
            obj *= scores_logits[:idx.shape[0]]
            obj = obj.repeat(int(self.select_num/obj.shape[0]),1)[:self.select_num] # repeat in case that too few object selected

        glob = glob.expand(obj.shape[0], 1024, 7, 7)
        x = torch.cat((obj, glob), dim=1) # (num, 2048+1024, 7, 7)

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        # print('x.shape:', x.shape)
        tmp = torch.flatten(x)
        x = self.fc1(tmp)
        if self.side:
            side = self.fc_side(tmp)

        return (x, side) if self.side else x


class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.conv1 = nn.Conv2d(2048, 256, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 1, 4)

    def forward(self, x):
        # x.shape (N, 2048, 7, 7)
        #print(x.shape)
        weights = self.relu1(self.conv1(x))
        weights = self.conv2(weights) # (N, 1, 1, 1)

        return weights

def build_predictor(cfg, side):
    model = Predictor(cfg, side=side)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roi = torch.randn((15,2048,7,7))
    glob = torch.randn((1,1024,14,14))
    roi = roi.to(device)
    glob = glob.to(device)
    x = {'roi_features':roi,
         'glob_feature':glob}

    model = Predictor()
    model.to(device)
    output = model(x)
