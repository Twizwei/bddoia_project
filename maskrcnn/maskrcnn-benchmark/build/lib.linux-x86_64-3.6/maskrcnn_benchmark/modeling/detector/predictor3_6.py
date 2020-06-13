
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

        self.fc_glob = nn.Linear(2048, 256)
        self.relu_glob = nn.ReLU()
        self.fc_obj = nn.Linear(2048, 256)
        self.relu_obj = nn.ReLU()

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
        # print(glob.shape)

        # Processing object bounding box
        bbox = x['bbox']# (N, 4)
        # print(bbox)
        obj = x['roi_features'] # (N, 1024, 14, 14)

        # Select objects
        scale_ratio = (720//glob.shape[2], 1280//glob.shape[3])
        scores = self.selector(glob, bbox, scale_ratio)
        scores, idx = torch.sort(scores, dim=0, descending=True)
        # print(scores)
        
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


        # Processing object features
        obj = self.head_obj(obj) # (select_num, 2048, 7, 7)
        obj = self.avgpool(obj) # (select_num, 2048, 1, 1)
        obj = obj.squeeze() # (select_num, 2048)

        # Processing object bounding box
        # (xyxy) to (xywh)
        # bbox[:,2] = torch.log(torch.div(640,(bbox[:,2]-bbox[:,0])))
        # bbox[:,3] = torch.log(torch.div(360,(bbox[:,3]-bbox[:,1])))
        # bbox[:,0] = torch.div(bbox[:,0],640)
        # bbox[:,1] = torch.div(bbox[:,1],360)
        # print(bbox)
        bbox[:,2] = torch.log(torch.div(1280,(bbox[:,2]-bbox[:,0])))
        bbox[:,3] = torch.log(torch.div(720,(bbox[:,3]-bbox[:,1])))
        bbox[:,0] = torch.div(bbox[:,0],1280)
        bbox[:,1] = torch.div(bbox[:,1],720)

        # Processing global feature
        glob = self.avgpool(glob) # (1, 2048, 1, 1)
        glob = glob.squeeze().unsqueeze(0) # (1, 2048)

        glob = self.relu_glob(self.fc_glob(glob))
        obj = self.relu_obj(self.fc_obj(obj))
        obj = torch.cat((bbox, obj), dim=1) # (select_num, 260)

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
        self.pooler = Pooler()

    def forward(self, x, bbox, scale_ratio):
        scores = self.pooler(x, bbox, scale_ratio)

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
