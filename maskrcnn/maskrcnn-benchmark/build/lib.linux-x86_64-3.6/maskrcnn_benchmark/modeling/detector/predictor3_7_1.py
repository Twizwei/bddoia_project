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
        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=7)

        # Layers for object features
        self.avgpool_obj = nn.AdaptiveAvgPool2d(output_size=1)

        # Selector
        self.random_select = random_select
        self.select_num = select_num
        if not random_select:
            self.selector = Selector()
            self.sftmax = nn.Softmax(dim=0)

        self.avgpool_glob2 = nn.AdaptiveAvgPool2d(output_size=1)

        # FC
        self.drop = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear((256+256), 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(select_num*256, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, class_num)
        if self.side:
            self.fc_side = nn.Linear(256*select_num, 21)



    def forward(self, x):
        # Processing global feature
        glob = self.relu1(self.conv1(x['glob_feature']))  # (1, 512, 45, 80)
        glob = self.relu2(self.conv2(glob))  # (1, 256, 45, 80)
        glob = self.avgpool_glob(glob) # (1, 256, 7, 7)
        # Processing object features
        obj = x['roi_features'] # (N, 1024, 14, 14)
        obj = self.relu1(self.conv1(obj)) 
        obj = self.relu2(self.conv2(obj)) # (N, 256, 14, 14)
        obj = self.avgpool_obj(obj) # (N, 256, 1, 1)
        obj = obj.expand(obj.shape[0], 256, 7, 7)   
        
        if not self.random_select:
            glob_expand = glob.expand(obj.shape[0], 256, 7, 7)
            x = torch.cat((obj, glob_expand), dim=1) # (N, 256+256, 7, 7)
            # Select objects
            scores = self.selector(x)
            # print(torch.flatten(scores))
            scores, idx = torch.sort(scores, dim=0, descending=True)
            scores_logits = self.sftmax(scores)
        
            if self.select_num <= idx.shape[0]: # in case that too few objects detected
                idx = idx[:self.select_num].reshape(self.select_num)
                obj = obj[idx]
                # obj /= self.select_num
                obj *= scores_logits[:self.select_num] # shape(select_num, 256, 1, 1)
                # print(x.shape)
            else:
                idx = idx.reshape(idx.shape[0])
                obj = obj[idx]
                # obj /= self.select_num
                obj *= scores_logits[:idx.shape[0]] # shape(select_num, 256, 1, 1)
                print(obj.shape)
                obj = obj.repeat(int(self.select_num/obj.shape[0]) + 1, 1, 1, 1)[:self.select_num] # repeat in case that too few object selected
                print(obj.shape)

            del scores, idx, x, glob_expand
        else:
            idx = torch.randperm(obj.shape[0])[:self.select_num]
            obj = obj[idx]
            obj *= 1/obj.shape[0] # shape(select_num, 256, 7, 7)

        obj = self.avgpool_obj(obj) # (select_num, 256, 1, 1)
        glob = self.avgpool_glob2(glob) # (1, 256, 1, 1)
        glob = glob.expand(obj.shape[0], 256, 1, 1) # (select_num, 256, 1, 1)
        
        x = torch.cat((obj, glob), dim=1) # (select_num, 256+256, 1, 1)
        x = torch.squeeze(x) # (select_num, 256+256)
        del obj, glob
        
        x = self.drop(self.relu1(self.fc1(x))) # (select_num, 256)
        tmp = torch.flatten(x) # (select_num * 256)
        x = self.drop(self.fc2(tmp)) # (64)
        x = self.drop(self.fc3(x)) # (class_num)
        if self.side:
            side = self.fc_side(tmp) # (21)

        return (x, side) if self.side else x



class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.conv1 = nn.Conv2d(256+256, 256, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 64, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x)) # (N, 256, 7, 7)
        x = self.relu2(self.conv2(x)) # (N, 64, 7, 7)
        x = self.relu3(self.conv3(x)) # (N, 1, 7, 7)
        x = self.avgpool(x) #  (N, 1, 1, 1)
        return x

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
