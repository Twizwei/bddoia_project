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
from tensorboardX import SummaryWriter

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model

from DataLoader_together import BatchLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def train(cfg, args):

    # Initialize the network
    model = build_detection_model(cfg)

    print("Load from checkpoint?", bool(args.from_checkpoint))
    if not bool(args.from_checkpoint):
        path = args.model_root
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint, strict=False)
    else:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
    # model.train()

    print("Freeze faster rcnn?", bool(args.freeze))
    for i, child in enumerate(model.children()):
        # print(i)
        # print(child)
        if i < 3:
            for param in child.parameters():
                param.requires_grad = False
                # param.requires_grad = not (bool(args.freeze))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outdir = cfg.OUTPUT_DIR
    writer = SummaryWriter(outdir+'tblogs')
    

    class_weights = [1, 1, 2, 2]
    w = torch.FloatTensor(class_weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    criterion2 = nn.BCEWithLogitsLoss().cuda()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(args.initLR), weight_decay=float(args.weight_decay))
    # optimizer = optim.SGD(model.parameters(), lr=float(args.initLR), momentum=0.9, weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize DataLoader
    Dataset = BatchLoader(
        imageRoot=args.imageroot,
        gtRoot=args.gtroot,
        reasonRoot=args.reasonroot,
        cropSize=(args.imHeight, args.imWidth)
    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=24, shuffle=True)
    len_data = len(dataloader)

    for epoch in range(0, args.num_epoch):
        trainingLog = open(outdir + ('trainingLogTogether_{0}.txt'.format(epoch)), 'w')
        trainingLog.write(str(args))
        lossArr = []
        AccuracyArr = []
        AccSideArr = []
        for i, dataBatch in enumerate(dataloader):

            # Read data
            img_cpu = dataBatch['img']
            imBatch = img_cpu.to(device)

            target_cpu = dataBatch['target']
            targetBatch = target_cpu.to(device)
            # ori_img_cpu = dataBatch['ori_img']

            # log to TFboard
            niter = epoch * len_data + i
            

            if cfg.MODEL.SIDE:
                reason_cpu = (dataBatch['reason']).type(torch.FloatTensor)
                reasonBatch = reason_cpu.to(device)

            optimizer.zero_grad()
            if cfg.MODEL.SIDE:
                # pred, pred_reason = model(imBatch)
                pred_reason = model(imBatch)
                # Joint loss
                # loss1 = criterion(pred, targetBatch)
                # writer.add_scalar('Train/Loss1', loss1.cpu().data.item(), niter)
                if args.shrink_loss:
                    loss2 = shrink_pred(pred_reason, targetBatch, reasonBatch, criterion2)
                else:
                    loss2 = criterion2(pred_reason, reasonBatch)
                writer.add_scalar('Train/Loss2', loss2.cpu().data.item(), niter)

                # loss = loss1 + loss2
                loss = loss2
                writer.add_scalar('Train/Loss', loss2.cpu().data.item(), niter)

                # loss1 = criterion(pred, targetBatch)
                # idx = reasonBatch > 0

            else:
                pred = model(imBatch)
                loss = criterion(pred, targetBatch)

            # torch.cuda.empty_cache()
            # pred, selected_boxes = model(imBatch)
            # DrawBbox(ori_img_cpu[0], selected_boxes[0])
            # plt.clf()
            # plt.close()

            # print(pred)
            # print(targetBatch)

            loss.backward()
            optimizer.step()
            loss_cpu = loss.cpu().data.item()

            

            lossArr.append(loss_cpu)
            meanLoss = np.mean(np.array(lossArr))

            # Calculate accuracy
            # predict = torch.sigmoid(pred) > 0.5

            # f1 = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
            # AccuracyArr.append(f1)
            # meanAcc = np.mean(np.array(AccuracyArr))

            # log to TFborad
            # writer.add_scalar('Train/Acc', f1, niter)

            if cfg.MODEL.SIDE:
                predict_reason = torch.sigmoid(pred_reason) > 0.5
                f1_side = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')
                AccSideArr.append(f1_side)

            if i % 50 == 0:
                
                # print('prediction logits:', pred)
                # print('ground truth:', targetBatch.cpu().data.numpy())
                print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                    epoch, i, loss_cpu, meanLoss))
                trainingLog.write('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f \n' % (
                    epoch, i, loss_cpu, meanLoss))
                # print('Epoch %d Iteration %d Action Prediction: F1 %.5f Accumulated F1 %.5f' % (
                    # epoch, i, AccuracyArr[-1], meanAcc))
                if cfg.MODEL.SIDE:
                    meanAccSide = np.mean(AccSideArr)
                    print('Epoch %d Iteration %d Side Task: F1 %.5f Accumulated F1 %.5f' % (
                        epoch, i, AccSideArr[-1], meanAccSide))

                print("Selector Output:")
                # print(torch.flatten(hook_scores.output.data))
                
        scheduler.step()

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), (outdir + 'net_%d.pth' % (epoch + 1)))
        # if args.val and epoch % 10 == 0:
        #    print("Validation...")
        #    run_test(cfg, args)
    print("Saving final model...")
    torch.save(model.state_dict(), (outdir + 'net_Final.pth'))
    writer.export_scalars_to_json(os.path.join(outdir, 'scalars.json'))
    print("Done!")

def shrink_pred(pred_reason, actionTarget, reasonTarget, criterion):
    loss = torch.tensor([0], dtype=torch.float32).cuda()
    pred = []
    target = []
    if actionTarget.shape[1] == 4:
        for i in range(actionTarget.shape[0]):
            if actionTarget[i][0]:
                pred.append(pred_reason[i][0:3])
                target.append(reasonTarget[i][0:3])
            if actionTarget[i][1]:
                pred.append(pred_reason[i][3:9])
                target.append(reasonTarget[i][3:9])
            if actionTarget[i][2]:
                pred.append(pred_reason[i][9:12])
                pred.append(pred_reason[i][15:18])
                target.append(reasonTarget[i][9:12])
                target.append(reasonTarget[i][15:18])
            if actionTarget[i][3]:
                pred.append(pred_reason[i][12:15])
                pred.append(pred_reason[i][18:21])
                target.append(reasonTarget[i][12:15])
                target.append(reasonTarget[i][18:21])
    elif actionTarget.shape[1] == 6:
        for i in range(actionTarget.shape[0]):
            if actionTarget[i][0]:
                pred.append(pred_reason[i][0:3])
                target.append(reasonTarget[i][0:3])
            if actionTarget[i][1]:
                pred.append(pred_reason[i][3:9])
                target.append(reasonTarget[i][3:9])
            if actionTarget[i][2]:
                pred.append(pred_reason[i][15:18])
                target.append(reasonTarget[i][15:18])
            if actionTarget[i][3]:
                pred.append(pred_reason[i][18:21])
                target.append(reasonTarget[i][18:21])
            if actionTarget[i][4]:
                pred.append(pred_reason[i][9:12])
                target.append(reasonTarget[i][9:12])
            if actionTarget[i][5]:
                pred.append(pred_reason[i][12:15])
                target.append(reasonTarget[i][12:15])

    if len(pred) == 0:
        return torch.tensor(0).cuda()
    pred = torch.cat(pred).cuda()
    target = torch.cat(target).cuda()
    loss = criterion(pred, target)
    return loss

def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="../maskrcnn-benchmark/configs/baseline.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0001,
        help="Weight decay",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--initLR",
        help="Initial learning rate",
        default=0.001
    )
    parser.add_argument(
        "--freeze",
        default=False,
        help="If freeze faster rcnn",
    )
    parser.add_argument(
        "--imageroot",
        type=str,
        help="Directory to the images",
        default="../maskrcnn/data/lastframe/data"
    )
    parser.add_argument(
        "--gtroot",
        type=str,
        help="Directory to the groundtruth",
        default="../maskrcnn/data/lastframe/data/train_25k_images_actions.json",

    )
    parser.add_argument(
        "--reasonroot",
        type=str,
        help="Directory to the explanations",
        default="../maskrcnn/data/lastframe/data/train_25k_images_reasons.json"
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
        default=2
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Give this experiment a name",
        default=str(datetime.datetime.now())
    )
    parser.add_argument(
        "--model_root",
        type=str,
        help="Directory to the trained model",
        default="."
    )
    parser.add_argument(
        "--shrink_loss",
        action='store_true',
        default=False,
        help='whether shrink loss2'
    )
    parser.add_argument(
        "--num_epoch",
        default=50,
        help="The number of epoch for training",
        type=int
    )
    parser.add_argument(
        "--from_checkpoint",
        default=False,
        help="If we need load weights from checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=".",
        help="The path to the checkpoint weights.",
        type=str,
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

    train(cfg, args)


if __name__ == "__main__":
    main()
