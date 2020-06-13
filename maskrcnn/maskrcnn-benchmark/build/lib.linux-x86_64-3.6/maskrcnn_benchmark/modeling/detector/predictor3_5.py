
import torch
import torch.nn as nn

from torchvision.models import resnet50
# from maskrcnn_benchmark.modeling.backbone import resnet


class Predictor(nn.Module):

    def __init__(self, cfg=None, select_num=10, class_num=4, side=False):
        super(Predictor, self).__init__()
        self.side = side

        self.head_glob = resnet50().layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.head_obj = resnet50().layer4

        self.fc1 = nn.Linear(2048, 256)
        self.relu1 = nn.ReLU()

        # Selector
        self.selector = Selector()
        self.select_num = select_num


        # FC
        self.drop = nn.Dropout(p=0.2) 
        self.fc2 = nn.Linear((select_num+1)*260, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, class_num)
        if self.side:
            self.fc_side = nn.Linear(128, 21)



    def forward(self, x):
        # Processing global feature
        glob = self.head_glob(x['glob_feature']) 
        glob = self.avgpool(glob) # (1, 2048, 1, 1)
        glob = glob.squeeze().unsqueeze(0) # (1, 2048)

        # Processing object features
        obj = self.head_obj(x['roi_features']) # (N, 2048, 7, 7)
        obj = self.avgpool(obj) # (N, 2048, 1, 1)
        obj = obj.squeeze() # (N, 2048)

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

        x = torch.cat((glob, obj), dim=0)
        x = self.relu1(self.fc1(x))
        glob = x[0].unsqueeze(0) # (1, 256)
        obj = x[1:] # (N, 256)

        obj = torch.cat((bbox, obj), dim=1) # (N, 260)

        # Select objects
        scores = self.selector(obj)
        scores, idx = torch.sort(scores, dim=0, descending=True)
        
        # score_logits = nn.functional.softmax(scores.squeeze())
        # print(score_logits)
        # scores_logits = self.sftmax(scores)
        # print(scores_logits)
        
        if self.select_num <= idx.shape[0]: # in case that too few objects detected
            idx = idx[:self.select_num].squeeze()
            obj = obj[idx]
            bbox = bbox[idx]
            obj /= self.select_num # (N, 260)
            # obj *= scores_logits[:self.select_num] # shape(select_num, 2048)
            # print(x.shape)
        else:
            idx = idx.squeeze()
            obj = obj[idx]
            bbox = bbox[idx]
            # obj *= scores_logits[:idx.shape[0]] # shape(idx.shape[0], 2048)
            obj /= self.select_num
            print(obj.shape)
            obj = obj.repeat(int(self.select_num/obj.shape[0]) + 1, 1)[:self.select_num] # repeat in case that too few object selected
            bbox = bbox.repeat(int(self.select_num/obj.shape[0]) + 1, 1)[:self.select_num]
            print(obj.shape)
        #print(scores)

        glob = torch.cat((torch.cuda.FloatTensor([[0,0,0,0]]), glob), dim=1) # (1, 4+256)
        x = torch.cat((glob, obj), dim=0) # (select_num+1, 260)
        # print(glob.shape)
        # print(obj.shape)

        del glob, obj, scores, idx

        x = torch.flatten(x) # ((select_num+1)*260)
        # print(x.shape)
        tmp = self.drop(self.relu2(self.fc2(x)))
        x = self.fc3(tmp)
        
        if self.side:
            side = self.fc_side(tmp)

        return (x, side) if self.side else x



class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.fc1 = nn.Linear(4+256, 32)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 1)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        # x.shape (N, 516)
        x = self.drop(self.relu1(self.fc1(x)))
        x = self.fc2(x) # (N, 1)

        return x

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
