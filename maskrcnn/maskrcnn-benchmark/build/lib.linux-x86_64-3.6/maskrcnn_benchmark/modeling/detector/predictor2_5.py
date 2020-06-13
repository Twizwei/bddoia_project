
import torch
import torch.nn as nn

# from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, cfg=None, select_num=10, class_num=4, side=False, random_select=False):
        super(Predictor, self).__init__()
        self.side = side

        # Layers for global feature
        self.avgpool_glob1 = nn.AdaptiveAvgPool2d(output_size=14)
        self.conv_glob1 = nn.Conv2d(1024, 512, 4, stride=2)
        self.relu_glob1 = nn.ReLU(inplace=True)
        self.conv_glob2 = nn.Conv2d(512, 256, 4)
        self.relu_glob2 = nn.ReLU(inplace=True)

        self.avgpool_glob2 = nn.AdaptiveAvgPool2d(output_size=1)

        # Layers for object features
        self.avgpool_obj = nn.AdaptiveAvgPool2d(output_size=1)

        # Selector
        self.selector = Selector()
        self.select_num = select_num
        self.sftmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.k = 1e6

        # FC
        self.drop = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear(2048+256, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, class_num)
        if self.side:
            self.fc_side = nn.Linear(256, 21)



    def forward(self, x):
        # Processing global feature
        glob = self.avgpool_glob1(x['glob_feature']) # (1, 1024, 14, 14)
        glob = self.relu_glob1(self.conv_glob1(glob))
        glob = self.relu_glob2(self.conv_glob2(glob)) 
        glob = self.avgpool_glob2(glob) # (1, 256, 1, 1)
        # Processing object features
        obj = x['roi_features'] # (N, 2048, 7, 7)

        glob2 = glob.expand(obj.shape[0], 256, 7, 7)
        x = torch.cat((obj, glob2), dim=1) # (N, 2048+256, 7, 7)

        # Select objects
        scores = self.selector(x) 
        scores = self.sftmax(scores) # (N,1,1,1)
        scores_sorted, idx = torch.sort(scores, dim=0, descending=True)
        
        if self.select_num < idx.shape[0]: # in case that too few objects detected
            score_thresh = (scores_sorted[self.select_num - 1] + scores_sorted[self.select_num])/2
            one_hot = self.sigmoid((scores - score_thresh) * self.k) > 0
            obj = (scores * one_hot.type_as(scores)) * obj # (N, 2048, 7, 7)
            obj = obj.sum(dim=0).unsqueeze(0) # (1, 2048, 7, 7)

        else:
            obj = obj.repeat(int(self.select_num/obj.shape[0]) + 1, 1, 1, 1)[:self.select_num] # repeat in case that too few object selected
            obj *= scores.repeat(int(self.select_num/obj.shape[0]) + 1, 1, 1, 1)[:self.select_num] # (N, 2048, 7, 7)
            obj = obj.sum(dim=0).unsqueeze(0)


        del scores, idx, x, glob2

        obj = self.avgpool_obj(obj) # (1, 2048, 1, 1)
        glob = self.avgpool_glob2(glob) # (1, 256, 1, 1)
        
        x = torch.cat((obj, glob), dim=1) # (1, 2048+256, 1, 1)
        x = x.squeeze(-1).squeeze(-1) # (1, 2048+256)
        del obj, glob
        
        tmp = self.drop(self.relu1(self.fc1(x))) # (1, 256)
        
        x = self.fc2(tmp) # (1, 64)
        x = self.fc3(x) # (1, class_num)
        if self.side:
            side = self.fc_side(tmp)

        return (x, side) if self.side else x



class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.conv1 = nn.Conv2d(2048+256, 256, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 16, 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 1, 3)

    def forward(self, x):
        # x.shape (N, 2048+256, 7, 7)
        weights = self.relu1(self.conv1(x)) # (N, 256, 5, 5)
        weights = self.relu2(self.conv2(weights)) # (N, 16, 3, 3)
        weights = self.conv3(weights) # (N, 1, 1, 1)

        return weights

def build_predictor(cfg, side, random_select):
    model = Predictor(cfg, side=side, random_select=random_select)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roi = torch.randn((15,2048,7,7))
    glob = torch.randn((1,1024,45,80))
    roi = roi.to(device)
    glob = glob.to(device)
    x = {'roi_features':roi,
         'glob_feature':glob}

    model = Predictor()
    model.to(device)
    output = model(x)
