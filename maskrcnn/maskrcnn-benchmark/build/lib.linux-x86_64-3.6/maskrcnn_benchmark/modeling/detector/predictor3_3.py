import torch
import torch.nn as nn

# from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, cfg=None, select_num=5, class_num=4, side=False, random_select=False):
        super(Predictor, self).__init__()
        self.side = side

        # Layers for global feature
        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=7)
        self.conv_glob1 = nn.Conv2d(1024, 512, 4, stride=2, padding=1)
        self.relu_glob1 = nn.ReLU()
        self.conv_glob2 = nn.Conv2d(512, 256, 4, stride=2, padding=1)
        self.relu_glob2 = nn.ReLU()

        self.avgpool_glob2 = nn.AdaptiveAvgPool2d(output_size=1)

        # Layers for object features
        self.conv_obj1 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.relu_obj1 = nn.ReLU()
        self.conv_obj2 = nn.Conv2d(1024, 256, 3, padding=1)
        self.relu_obj2 = nn.ReLU()
        self.avgpool_obj = nn.AdaptiveAvgPool2d(output_size=1)

        # Selector
        self.random_select = cfg.MODEL.RANDOM_SELECT
        self.select_num = select_num
        if not random_select:
            self.selector = Selector()
            # self.sftmax = nn.Softmax(dim=0)
            self.T = cfg.MODEL.TEMP

        # FC
        self.drop = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear(260+256, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256*select_num, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, class_num)
        if self.side:
            self.fc_side = nn.Linear(256*select_num, 21)



    def forward(self, x):
        # Processing global feature
        glob = self.relu_glob1(self.conv_glob1(x['glob_feature']))
        glob = self.relu_glob2(self.conv_glob2(glob))
        glob = self.avgpool_glob(glob) # (1, 256, 7, 7)
        # Processing object features
        obj = x['roi_features'] # (N, 2048, 7, 7)
        obj = self.relu_obj1(self.conv_obj1(obj))
        obj = self.relu_obj2(self.conv_obj2(obj)) # (N, 256, 7, 7)
        
        bbox = x['bbox'] # (N, 4)
        bbox[:,2] = torch.log(torch.div(1280, (bbox[:,2]-bbox[:,0])))
        bbox[:,3] = torch.log(torch.div(720, (bbox[:,3]-bbox[:,1])))
        bbox[:,0] = torch.div(bbox[:,0], 1280)
        bbox[:,1] = torch.div(bbox[:,1], 720)  
        bbox = bbox.unsqueeze(-1).unsqueeze(-1)
        bbox = bbox.expand(bbox.shape[0], 4, 7, 7) # (N, 4, 7, 7)

        # concat bbox and obj features
        obj = torch.cat((bbox, obj), dim=1) # (N, 260, 7, 7)
        
        if not self.random_select:
            glob2 = glob.expand(obj.shape[0], 256, 7, 7)
            x = torch.cat((obj, glob2), dim=1) # (N, 256+260, 7, 7)
            # Select objects
            scores = self.selector(x)
            scores, idx = torch.sort(scores, dim=0, descending=True)
            # print(idx.shape)
            # scores_logits = self.sftmax(scores/self.T)
            # print(scores)
            # print(scores_logits)
        
            if self.select_num <= idx.shape[0]: # in case that too few objects detected
                idx = idx[:self.select_num].squeeze()
                obj = obj[idx]
                obj /= self.select_num
                # print(x.shape)
            else:
                idx = idx.reshape(idx.shape[0])
                obj = obj[idx]
                obj /= self.select_num
                print(obj.shape)    
                obj = obj.repeat(int(self.select_num/obj.shape[0]) + 1, 1, 1, 1)[:self.select_num] # repeat in case that too few object selected
                print(obj.shape)

            del scores, idx, x, glob2
        else:
            idx = torch.randperm(obj.shape[0])[:self.select_num]
            obj = obj[idx]
            obj *= 1/obj.shape[0] # shape(select_num, 2048, 7, 7)

        obj = self.avgpool_obj(obj) # (select_num, 260, 1, 1)
        glob = self.avgpool_glob2(glob) # (1, 256, 1, 1)
        glob = glob.expand(obj.shape[0], 256, 1, 1)
        
        x = torch.cat((obj, glob), dim=1) # (select_num, 260+256, 1, 1)
        x = torch.squeeze(x) # (select_num, 260+256)
        del obj, glob
        
        x = self.drop(self.relu1(self.fc1(x))) # (select_num, 256)
        
        
        tmp = torch.flatten(x) # (select_num * 256)
        x = self.drop(self.relu2(self.fc2(tmp))) # 0929
        # x = self.drop(self.fc2(tmp)) # 0924
        x = self.drop(self.fc3(x))
        # print(tmp.shape)
        # x = self.fc2(tmp)
        # x = self.fc3(x)
        if self.side:
            side = self.fc_side(tmp)

        return (x, side) if self.side else x



class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.conv1 = nn.Conv2d(256+260, 256, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 16, 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 1, 3)

    def forward(self, x):
        # x.shape (N, 260+256, 7, 7)
        x = self.relu1(self.conv1(x)) # (N, 256, 5, 5)
        x = self.relu2(self.conv2(x)) # (N, 16, 3, 3)
        # print(x.shape)
        x = self.conv3(x) # (N, 1, 1, 1)
        # print(x.shape)

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
