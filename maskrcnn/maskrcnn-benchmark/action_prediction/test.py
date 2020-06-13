import argparse
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.detector import build_detection_model

from DataLoader_together import BatchLoader
# from Dataloader_single import BatchLoader
# from DataLoader_together6 import BatchLoader

from utils import attention
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def test(cfg, args):
    # torch.cuda.set_device(5)

    # Initialize the network
    model = build_detection_model(cfg)
    print(model)
    model.eval()
    #print(model)
    
    # model load weights
    model.load_state_dict(torch.load(args.model_root))
    # model.load_state_dict(torch.load(cfg.MODEL.WEIGHT))

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    outdir = os.path.join(cfg.OUTPUT_DIR,'inference/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    # Initialize DataLoader
    Dataset = BatchLoader(
        imageRoot = args.imageroot,
        gtRoot = args.gtroot,
        reasonRoot = args.reasonroot,
        cropSize = (args.imHeight, args.imWidth)
    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=24, shuffle=False)

    AccOverallArr = []
    TargetArr = []
    PredArr = []
    RandomArr = []
    RandomF1 = []
    
    AccOverallReasonArr = []
    TargetReasonArr = []
    PredReasonArr = []
    RandomReasonArr = []
    
    

    SaveFilename = (outdir + 'TestingLog.txt')
    TestingLog = open(SaveFilename, 'w')
    print('Save to ', SaveFilename)
    TestingLog.write(str(args) + '\n')


    count = dataloader.__len__()
    for i, dataBatch in enumerate(dataloader): 
        print('Finished: {} / {}'.format(i, count))
        print('Finished: %.2f%%' % (i /count * 100))
        # Read data
        with torch.no_grad():
            img_cpu = dataBatch['img']
            imBatch = img_cpu.to(device)
            ori_img_cpu = dataBatch['ori_img']

            target_cpu = dataBatch['target']
            targetBatch = target_cpu.to(device)
            if cfg.MODEL.SIDE:
                reason_cpu = dataBatch['reason']
                reasonBatch = reason_cpu.to(device)

                pred, pred_reason = model(imBatch)

            else:
                pred = model(imBatch)

        # Calculate accuracy
        predict = torch.sigmoid(pred) > 0.5
        TargetArr.append(target_cpu.data.numpy())
        PredArr.append(predict.cpu().data.numpy())
        # print(predict)
        # print(target_cpu)
        f1_overall = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
        AccOverallArr.append(f1_overall)
        
        # random guess
        random = np.random.randint(0,2,(predict.shape[0], predict.shape[1]))
        RandomArr.append(random)
        f1_random = f1_score(target_cpu.data.numpy(), random, average='samples')
        RandomF1.append(f1_random)
        
        if cfg.MODEL.SIDE:
            predict_reason = torch.sigmoid(pred_reason) > 0.5
            TargetReasonArr.append(reason_cpu.data.numpy())
            PredReasonArr.append(predict_reason.cpu().data.numpy())
            f1_overall = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')
            AccOverallReasonArr.append(f1_overall)
            
            # random guess
            random = np.random.randint(0,2,(predict_reason.shape[0], predict_reason.shape[1]))
            RandomReasonArr.append(random)

        


        print('prediction logits:', pred)
        print('prediction action: \n {}'.format(predict))
        print('ground truth: \n', targetBatch.cpu().data.numpy())
        print('Accumulated Overall Action acc: ', np.mean(AccOverallArr))

        TestingLog.write('Iter ' + str(i) + '\n')
        TestingLog.write('prediction logits:' + str(pred) + '\n')
        TestingLog.write('prediction action: \n {}'.format(predict) + '\n')
        TestingLog.write('ground truth: \n' + str(targetBatch.cpu().data.numpy()) + '\n')
        if cfg.MODEL.SIDE:
            print('prediction reason: \n {}'.format(predict_reason))
            print('ground truth: \n', reason_cpu.data.numpy())
            print('Accumulated Overall Reason acc: ', np.mean(AccOverallReasonArr))
            
            TestingLog.write('prediction reason: \n {}'.format(predict_reason) + '\n')
            TestingLog.write('ground truth: \n' + str(reason_cpu.data.numpy()) + '\n')

        TestingLog.write('\n')


    TargetArr = List2Arr(TargetArr)
    PredArr = List2Arr(PredArr)
    RandomArr = List2Arr(RandomArr)
    
    f1_macro, f1_micro = ComputeClsAcc(TargetArr, PredArr)

    print(TargetArr.shape)
    print(PredArr.shape)

    # np.save(outdir+'Target.npy', TargetArr)
    # np.save(outdir+'Pred.npy', PredArr)

    f1_pred = f1_score(TargetArr, PredArr, average=None)
    f1_rand = f1_score(TargetArr, RandomArr, average=None)

    # print("Random guess acc:{}".format(np.mean(np.array(RandomAcc),axis=0)))
    print("Action Random guess acc:{}".format(f1_rand))
    print("Action Random guess overall acc:{}".format(np.mean(RandomF1)))
    print("Action Category Acc:{}".format(f1_pred))
    print("Action Average Acc:{}".format(np.mean(f1_pred)))
    print("Action Overall acc:{}".format(np.mean(np.array(AccOverallArr), axis=0)))

    print("Action f1 macro Acc:{}".format(f1_macro))
    print("Action mean f1 macro Acc:{}".format(np.mean(f1_macro)))
    print("Action f1 micro Acc:{}".format(f1_micro))
    print("Action mean f1 micro Acc:{}".format(np.mean(f1_micro)))

    TestingLog.write("Action Random guess acc:{}".format(f1_rand))
    TestingLog.write("Action Category Acc:{}".format(f1_pred))
    TestingLog.write("Action Average Acc:{}".format(np.mean(f1_pred)))
    TestingLog.write("Action Overall acc:{}".format(np.mean(np.array(AccOverallArr), axis=0)))
    TestingLog.write("Action f1 macro Acc:{}".format(f1_macro))
    TestingLog.write("Action mean f1 macro Acc:{}".format(np.mean(f1_macro)))
    TestingLog.write("Action f1 micro Acc:{}".format(f1_micro))
    TestingLog.write("Action mean f1 micro Acc:{}".format(np.mean(f1_micro)))
    
    if cfg.MODEL.SIDE:
        TargetReasonArr = List2Arr(TargetReasonArr)
        PredReasonArr = List2Arr(PredReasonArr)
        RandomReasonArr = List2Arr(RandomReasonArr)

        # np.save(outdir+'TargetReason.npy', TargetReasonArr)
        # np.save(outdir+'PredReason.npy', PredReasonArr)
        
        f1_pred_reason = f1_score(TargetReasonArr, PredReasonArr, average=None)
        f1_pred_rand = f1_score(TargetReasonArr, RandomReasonArr, average=None)

        f1_macro, f1_micro = ComputeClsAcc(TargetReasonArr, PredReasonArr)

        
        print("Reason Random guess acc:{}".format(f1_pred_rand))
        print("Reason Category Acc:{}".format(f1_pred_reason))
        print("Reason Average Acc:{}".format(np.mean(f1_pred_reason)))
        print("Reason Overall Acc:{}".format(np.mean(np.array(AccOverallReasonArr), axis=0)))
        print("Reason f1 macro Acc:{}".format(f1_macro))
        print("Reason mean f1 macro Acc:{}".format(np.mean(f1_macro)))
        print("Reason f1 micro Acc:{}".format(f1_micro))
        print("Reason mean f1 micro Acc:{}".format(np.mean(f1_micro)))

        TestingLog.write("Reason Random guess acc:{}".format(f1_pred_rand))
        TestingLog.write("Reason Category Acc:{}".format(f1_pred_reason))
        TestingLog.write("Reason Average Acc:{}".format(np.mean(f1_pred_reason)))
        TestingLog.write("Reason Overall Acc:{}".format(np.mean(np.array(AccOverallReasonArr), axis=0)))
        TestingLog.write("Reason f1 macro Acc:{}".format(f1_macro))
        TestingLog.write("Reason mean f1 macro Acc:{}".format(np.mean(f1_macro)))
        TestingLog.write("Reason f1 micro Acc:{}".format(f1_micro))
        TestingLog.write("Reason mean f1 micro Acc:{}".format(np.mean(f1_micro)))
    
    
def List2Arr(List):
    Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
    Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])

    return np.vstack((Arr1, Arr2))

def ComputeClsAcc(target, pred):
    """
    target - target array, (N, cls)
    pred - prediction array (N, cls)
    """
    f1_macro = []
    f1_micro = []
    for cls in range(target.shape[1]):
        target_ = target[:, cls]
        pred_ = pred[:, cls]
        f1_macro.append(f1_score(target_, pred_, average='macro'))
        f1_micro.append(f1_score(target_, pred_, average='micro'))
    f1_macro = np.array(f1_macro)
    f1_micro = np.array(f1_micro)

    return f1_macro, f1_micro


def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="./maskrcnn-benchmark/configs/baseline.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    )
    parser.add_argument(
        "--is_cat",
        default=False,
        help="If we use concatenation on object features",
        type=bool,

    )
    parser.add_argument(
        "--side",
        default=False,
        help="If we use side task",
        type=bool,
    
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--imageroot",
        type=str,
        help="Directory to the images",
        default="./datasets/data25k/lastframe/"
    )
    parser.add_argument(
        "--gtroot",
        type=str,
        help="Directory to the groundtruth",
        default="./datasets/data25k/info/val_25k_image_action_6.json"
    )
    parser.add_argument(
        "--reasonroot",
        type=str,
        help="Directory to the reason gt",
        default="./datasets/data25k/info/val_25k_images_reasons.json"
    
    )
    parser.add_argument(
        "--imWidth",
        type=int,
        help="Crop to width",
        default=1280
    )
    parser.add_argument(
        "--imHeight",
        type=int,
        help="Crop to height",
        default=720
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size",
        default=1
    )
    parser.add_argument(
        "--model_root",
        type=str,
        help="Directory to the trained model",
        default="./"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to the trained model",
        default="./output/"

    )

    args = parser.parse_args()
    print(args)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if torch.cuda.is_available():
        print("CUDA device is available.")

    # output directory
    outdir = cfg.OUTPUT_DIR
    print("Save path:", outdir)
    if outdir:
        mkdir(outdir)

    #    logger = setup_logger("training", outdir)

    test(cfg, args)


if __name__ == "__main__":
    main()
