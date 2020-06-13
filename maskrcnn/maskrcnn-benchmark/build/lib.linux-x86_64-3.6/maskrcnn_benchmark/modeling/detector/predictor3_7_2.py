import torch
import torch.nn as nn

# from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, cfg=None, select_num=10, class_num=4, side=False, random_select=False):
        super(Predictor, self).__init__()
        self.side = side

        # Layers for global feature
        self.conv1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=7)

        # Layers for object features
        self.avgpool_obj = nn.AdaptiveAvgPool2d(output_size=1)

        # Selector
        self.random_select = random_select
        self.select_num = select_num
        if not random_select:
            self.selector = Selector(self.select_num)

        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=1)

        # FC
        self.drop = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear((256+256), 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(select_num*256, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, class_num)
        if self.side:
            self.fc_side = nn.Linear(256*select_num, 21)



    def forward(self, x):
        # Processing global feature
        glob = self.relu1(self.conv1(x['glob_feature']))  # (1, 512, 45, 80)
        glob = self.relu2(self.conv2(glob))  # (1, 256, 45, 80)

        # Processing object features
        obj = x['roi_features'] # (N, 1024, 14, 14)
        obj = self.relu1(self.conv1(obj)) 
        obj = self.relu2(self.conv2(obj)) # (N, 256, 14, 14)
        # obj = self.avgpool_obj(obj) # (N, 256, 1, 1)
        # obj = obj.expand(obj.shape[0], 256, 7, 7)   

        # processing bbox
        bbox = x['bbox']
        scale_ratio = (720//glob.shape[2], 1280//glob.shape[3])
        scores = self.selector(glob, bbox, scale_ratio)
        idx = scores > 0.49
        # idx = scores >= 0.5
        # print(idx.shape)
        # print(scores.shape)
        # scores = scores[idx.squeeze()]
        # obj = obj[idx.squeeze()]
        # obj = scores * obj
        obj = ( scores[idx.squeeze()] * obj[idx.squeeze()])[:self.select_num] # choose the top (select_num) objects
        # print("obj", obj.shape)
        
        if not self.random_select:
            if self.select_num > obj.shape[0]: # in case that too few objects detected
                # print(obj.shape)
                if obj.shape[0] == 0:
                    obj = x['roi_features'][:select_num]
                obj = obj.repeat(int(self.select_num/obj.shape[0]) + 1, 1, 1, 1)[:self.select_num] # repeat in case that too few object selected
                # print(obj.shape)
        else:
            idx = torch.randperm(obj.shape[0])[:self.select_num]
            obj = obj[idx]
            obj *= 1/obj.shape[0] # shape(select_num, 256, 7, 7)
        
        # print(obj.shape)
        obj = self.avgpool_obj(obj) # (select_num, 256, 1, 1)
        glob = self.avgpool_glob(glob) # (1, 256, 1, 1)
        glob = glob.expand(obj.shape[0], 256, 1, 1) # (select_num, 256, 1, 1)
        
        x = torch.cat((obj, glob), dim=1) # (select_num, 256+256, 1, 1)
        x = torch.squeeze(x) # (select_num, 256+256)
        # print(x.shape)
        del obj, glob
        
        x = self.drop(self.relu_fc1(self.fc1(x))) # (select_num, 256)
        tmp = torch.flatten(x) # (select_num * 256)

        # print(tmp.shape)
        x = self.drop(self.relu_fc2(self.fc2(tmp))) # (64)
        x = self.drop(self.fc3(x)) # (class_num)
        if self.side:
            side = self.fc_side(tmp) # (21)

        return (x, side) if self.side else x



class Selector(nn.Module):
    def __init__(self, select_num):
        super(Selector, self).__init__()
        self.select_num = select_num
        self.pooler = Pooler()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bbox, scale_ratio):
        scores = self.pooler(x, bbox, scale_ratio)
        # print(scores.shape)
        _, idx = torch.sort(scores, dim=0, descending=True)
        if idx.shape[0] > self.select_num:
            thresh = 0.5 * (scores[idx[self.select_num - 1]] + scores[idx[self.select_num]]).squeeze()
        else:
            thresh = 1.1 * (scores[idx[self.select_num - 1]]).squeeze()
        scores = self.sigmoid((scores - thresh) * 100)
        # print(scores.shape)
        return scores

class Pooler(nn.Module):
    def __init__(self, output_size=(1, 1, 1)):
        super(Pooler, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=output_size)

    def forward(self, feature, proposals, scale_ratio):
        scores = []
        # print(feature.shape)
        for i in range(proposals.shape[0]):
            bbox = proposals[i] # (x1, y1, x2, y2)
            # print(bbox)
            x1, y1, x2, y2 = bbox
            x1 = (x1//scale_ratio[1]).type(torch.int)
            x2 = (x2//scale_ratio[1]).type(torch.int)
            y1 = (y1//scale_ratio[0]).type(torch.int)
            y2 = (y2//scale_ratio[0]).type(torch.int)
            # print(x1, x2, y1, y2)
            score = self.avgpool(feature[:, :, y1:y2+1, x1:x2+1])
            scores.append(score)
            
        return torch.cat(scores)

def build_predictor(cfg, side, random_select=False):
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
