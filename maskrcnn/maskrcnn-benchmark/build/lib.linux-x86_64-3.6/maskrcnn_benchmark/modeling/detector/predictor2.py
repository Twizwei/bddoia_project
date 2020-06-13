import torch
import torch.nn as nn

# from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, cfg=None, select_num=15, class_num=4, side=False):
        super(Predictor, self).__init__()
        self.side = side

        # Layers for global feature
        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=7)
        self.conv_glob1 = nn.Conv2d(1024, 512, 4)
        self.relu_glob1 = nn.ReLU(inplace=True)
        self.conv_glob2 = nn.Conv2d(512, 256, 4)
        self.relu_glob2 = nn.ReLU(inplace=True)


        # Layers for object features
        # self.conv_obj1 = nn.Conv2d(2048, 2048, 4)
        # self.relu_obj1 = nn.ReLU(inplace=True)
        # self.conv_obj2 = nn.Conv2d(2048, 2048, 4)
        # self.relu_obj2 = nn.ReLU(inplace=True)
        self.avgpool_obj = nn.AdaptiveAvgPool2d(output_size=1)

        # Selector
        self.selector = Selector()
        self.select_num = select_num
        self.sftmax = nn.Softmax(dim=0)

        # FC
        # self.fc = nn.Linear((2048+256)*select_num, class_num)
        # if self.side:
        #    self.fc_side = nn.Linear((2048+256)*select_num, 21)
        self.drop = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear(2048+256, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256*select_num, class_num)
        if self.side:
            self.fc_side = nn.Linear(256*select_num, 21)



    def forward(self, x):
        # Processing global feature
        glob = self.avgpool_glob(x['glob_feature'])
        glob = self.relu_glob1(self.conv_glob1(glob))
        glob = self.relu_glob2(self.conv_glob2(glob))
        # Processing object features
        # obj = self.relu_obj1(self.conv_obj1(x['roi_features']))
        # obj = self.relu_obj2(self.conv_obj2(obj))
        obj = self.avgpool_obj(x['roi_features']) # (N, 2048, 1, 1)

        glob = glob.expand(obj.shape[0], 256, 1, 1)
        x = torch.cat((obj, glob), dim=1) # (N, 2048+256, 1, 1)
        x = torch.squeeze(x, dim=3)
        x = torch.squeeze(x, dim=2) # (N, 2048+256)

        # Select objects
        scores = self.selector(x)
        scores, idx = torch.sort(scores, dim=0, descending=True)
        scores_logits = self.sftmax(scores)
        # print(idx.shape[0])
        
        if self.select_num <= idx.shape[0]: # in case that too few objects detected
            idx = idx[:self.select_num].reshape(self.select_num)
            x = x[idx]
            x *= scores_logits[:self.select_num] # shape(select_num, 2048+256)
            # print(x.shape)
        else:
            idx = idx.reshape(idx.shape[0])
            x = x[idx]
            x *= scores_logits[:idx.shape[0]] # shape(idx.shape[0], 2048+256)
            
            x = x.repeat(int(self.select_num/x.shape[0]) + 1,1)[:self.select_num] # repeat in case that too few object selected
            print(x.shape)
        del obj, glob, scores, idx, scores_logits

        
#        tmp = torch.flatten(x)
#        x = self.fc(tmp)
#        if self.side:
#            side = self.fc_side(tmp)
#        
        
        x = self.drop(self.relu1(self.fc1(x)))
        tmp = torch.flatten(x)
        # x = self.drop(self.fc2(tmp))
        x = self.fc2(tmp)
        if self.side:
            side = self.fc_side(tmp)

        return (x, side) if self.side else x



class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.fc1 = nn.Linear(2048+256, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x.shape (N, 2048+256)
        weights = self.relu1(self.fc1(x))
        weights = self.fc2(weights) # (N, 1)

        return weights

def build_predictor(cfg, side):
    model = Predictor(cfg, side=side)
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
