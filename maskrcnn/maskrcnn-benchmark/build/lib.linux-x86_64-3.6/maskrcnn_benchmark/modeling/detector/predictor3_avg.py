
import torch
import torch.nn as nn

# from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, cfg=None, select_num=10, class_num=4, side=False):
        super(Predictor, self).__init__()
        self.side = side

        # Layers for global feature
        self.avgpool_glob1 = nn.AdaptiveAvgPool2d(output_size=7)
        self.conv_glob1 = nn.Conv2d(1024, 512, 4)
        self.relu_glob1 = nn.ReLU(inplace=True)
        self.conv_glob2 = nn.Conv2d(512, 256, 4)
        self.relu_glob2 = nn.ReLU(inplace=True)

        # Layers for object features
        # self.avgpool_obj = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_obj1 = nn.Conv2d(2048, 1000, 4)
        self.relu_obj1 = nn.ReLU(inplace=True)
        self.conv_obj2 = nn.Conv2d(1000, 256, 4)
        self.relu_obj2 = nn.ReLU(inplace=True)

        # Selector
        self.selector = Selector()
        self.select_num = select_num


        # FC
        self.drop = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear((select_num+1)*260, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, class_num)
        if self.side:
            self.fc_side = nn.Linear(128, 21)



    def forward(self, x):
        # Processing global feature
        glob = self.avgpool_glob1(x['glob_feature']) # (1, 1024, 7, 7)
        glob = self.relu_glob1(self.conv_glob1(glob))
        glob = self.relu_glob2(self.conv_glob2(glob)) # (1, 256, 1, 1)
        glob = torch.squeeze(glob)
        glob = torch.unsqueeze(glob, 0) # (1, 256)

        # Processing object features
        obj = x['roi_features'] # (N, 2048, 7, 7)
        obj = self.relu_obj1(self.conv_obj1(obj))
        obj = self.relu_obj2(self.conv_obj2(obj)) # (N, 256, 1, 1)
        obj = torch.squeeze(obj) # (N, 256)
        # Processing object bounding box
        bbox = x['bbox'] # (N, 4)
        # (xyxy) to (xywh)
        # bbox[:,2] = torch.log(torch.div(640,(bbox[:,2]-bbox[:,0])))
        # bbox[:,3] = torch.log(torch.div(360,(bbox[:,3]-bbox[:,1])))
        # bbox[:,0] = torch.div(bbox[:,0],640)
        # bbox[:,1] = torch.div(bbox[:,1],360)

        bbox[:,2] = torch.log(torch.div(1280,(bbox[:,2]-bbox[:,0])))
        bbox[:,3] = torch.log(torch.div(720,(bbox[:,3]-bbox[:,1])))
        bbox[:,0] = torch.div(bbox[:,0],1280)
        bbox[:,1] = torch.div(bbox[:,1],720)        
        obj2 = torch.cat((bbox, obj), dim=1) # (N, 4+256)

        glob2 = glob.expand(obj.shape[0], 256)
        x = torch.cat((obj2, glob2), dim=1) # (N, 260+256)
        #x = obj

        # Select objects
        scores = self.selector(x)
        scores, idx = torch.sort(scores, dim=0, descending=True)
        # print(scores_logits)
        
        if self.select_num <= idx.shape[0]: # in case that too few objects detected
            idx = idx[:self.select_num].reshape(self.select_num)
            obj = obj[idx]
            bbox = bbox[idx]
            obj /= self.select_num # shape(select_num, 2048)
            # print(x.shape)
        else:
            idx = idx.reshape(idx.shape[0])
            obj = obj[idx]
            bbox = bbox[idx]
            obj /= self.select_num # shape(idx.shape[0], 2048)
            print(obj.shape)
            obj = obj.repeat(int(self.select_num/obj.shape[0]) + 1, 1)[:self.select_num] # repeat in case that too few object selected
            bbox = bbox.repeat(int(self.select_num/obj.shape[0]) + 1, 1)[:self.select_num]
            print(obj.shape)
        #print(scores)

        del scores, idx, x, obj2, glob2

        
        glob = torch.cat((torch.cuda.FloatTensor([[0,0,0,0]]), glob),dim=1) # (1, 4+256)
        obj = torch.cat((bbox, obj), dim=1) # (N, 260)
        x = torch.cat((obj, glob), dim=0) # (N+1, 260)
        del obj, glob

        x = torch.flatten(x)
        tmp = self.drop(self.relu1(self.fc1(x)))
        x = self.fc2(tmp)
        
        if self.side:
            side = self.fc_side(tmp)

        return (x, side) if self.side else x



class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.fc1 = nn.Linear(4+256+256, 32)
        #self.fc1 = nn.Linear(260, 32)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 1)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        # x.shape (N, 516)
        weights = self.drop(self.relu1(self.fc1(x)))
        weights = self.fc2(weights) # (N, 1)

        return weights

def build_predictor(cfg, side):
    model = Predictor(cfg, side=side)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roi = torch.randn((15,2048,7,7))
    glob = torch.randn((1,1024,45,80))
    bbox = torch.randn((15, 4))
    roi = roi.to(device)
    glob = glob.to(device)
    bbox = bbox.to(device)
    x = {'roi_features':roi,
         'glob_feature':glob,
         'bbox':bbox}

    model = Predictor()
    model.to(device)
