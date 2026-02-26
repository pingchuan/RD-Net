import os
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import argparse
import datasets
from utils.common import generate_model, TwoStreamBatchSampler, set_seed
from utils.rd_loss import L2_loss, BceDiceLoss1, cont_loss, mse_consistency_loss, dpa, update_pseudo_labels, BceDiceLoss1_D
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--dataset', type=str, default='kvasir_SEG')
    parser.add_argument('--train_data_dir', type=str, default='kvasir_SEG/Train')
    parser.add_argument('--test_data_dir', type=str, default='kvasir_SEG/Test')
    #parse.add_argument('--train_data_dir', type=str, default='CVC-ClinicDB/Train')
    #parse.add_argument('--test_data_dir', type=str, default='CVC-ClinicDB/Test')
    

    # Training
    parser.add_argument('--checkpoints', type=str, default=r'./checkpoints')
    parser.add_argument('--shuffle', type=str, default=False)
    parser.add_argument('--method', type=str, default='rd_net')
    parser.add_argument('--expID', type=int, default=8888) # Kvasir-SEG:8888 CVC-ClinicDB: 10%:1888 2.5% 3888 Kvasir-ClinicDB:8888
    parser.add_argument('--batch_size', type=float, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model', type=str, default='ResNet34U_f')
    parser.add_argument('--ckpt_period', type=int, default=50)
    parser.add_argument('--total_iter', type=int, default=40000, help='Total number of training iterations')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mt', type=float, default=0.9)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--thresh_depth', type=float, default=0.85)
    parser.add_argument('--thresh', type=float, default=0.85)
    parser.add_argument('--thresh_dpa', type=float, default=0.95)

    # Labeling
    parser.add_argument('--label_mode', type=str, default='percentage')
    parser.add_argument('--labeled_bs', type=int, default=2)
    parser.add_argument('--labeled_perc', type=int, default=10)
    parser.add_argument('--labeled_num', type=int, default=10)


    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--ud', type=str, default=False)



    return parser.parse_args()





def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)



def train(args):
    set_seed(args.expID)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")

    # Load data
    train_data = getattr(datasets, args.dataset)(args.root, args.train_data_dir, mode='train')
    total_num = len(train_data)

    if args.label_mode == 'percentage':
        labeled_num = round(total_num * args.labeled_perc / 100)
        if labeled_num % 2 != 0:
            labeled_num -= 1
    else:
        labeled_num = args.labeled_num

    print(f"Total training images: {total_num}, labelled: {labeled_num} ({labeled_num / total_num * 100:.2f}%)")

    batch_sampler = TwoStreamBatchSampler(
        total_num, labeled_num, args.labeled_bs, int(args.batch_size) - args.labeled_bs, shuffle=args.shuffle
    )
    train_dataloader = DataLoader(train_data, batch_sampler=batch_sampler, shuffle=False, num_workers=args.num_workers)

    # === Compute nEpoch from total_iter ===
    iters_per_epoch = len(train_dataloader)
    if iters_per_epoch == 0:
        raise ValueError("Dataloader is empty!")
    args.nEpoch = math.ceil(args.total_iter / iters_per_epoch)
    #print(f"Total iterations: {args.total_iter} | Iters/epoch: {iters_per_epoch} => nEpoch: {args.nEpoch}")

    # Models
    model = generate_model(args).to(device)
    model_depth = generate_model(args).to(device)
    ema_model = generate_model(args, ema=True).to(device)

    # Optimizers & Schedulers
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)
    optimizer_depth = torch.optim.SGD(model_depth.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    #scheduler = LambdaLR(optimizer, lambda e: 1.0 - pow((e / args.nEpoch), args.power))
    #scheduler_depth = LambdaLR(optimizer_depth, lambda e: 1.0 - pow((e / args.nEpoch), args.power))

    scheduler = LambdaLR(optimizer, lambda step: 1.0 - pow((step / args.total_iter), args.power))
    scheduler_depth = LambdaLR(optimizer_depth, lambda step: 1.0 - pow((step / args.total_iter), args.power))
    # Losses
    criterion = BceDiceLoss1().to(device)
    criterion_d = BceDiceLoss1_D().to(device)

    # Setup checkpoint directory
    ckpt_dir = os.path.join(
        args.checkpoints,
        f"{args.dataset}",
        f"{args.method}_{args.model}",
        f"exp{args.expID}_{args.labeled_perc}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    global_iter = 0
    print('--- Start Training ---')

    for epoch in range(args.nEpoch):
        if global_iter >= args.total_iter:
            break

        model.train()
        model_depth.train()
        total_batch = int(labeled_num / args.labeled_bs)
        progress_bar = tqdm(enumerate(train_dataloader), total=total_batch)

        for batch_idx, data in progress_bar:
            if global_iter >= args.total_iter:
                break

            # Unpack data
            images, images_s, gts, depths, depth1s = data['image'], data['image_s'], data['label'], data['depth'], data['depth1']
            labeled_img = images[:args.labeled_bs].to(device)
            unlabeled_img = images[args.labeled_bs:].to(device)
            unlabeled_img_s = images_s[args.labeled_bs:].to(device)
            labeled_depth = depths[:args.labeled_bs].to(device)
            unlabeled_depth = depths[args.labeled_bs:].to(device)
            unlabeled_depth1 = depth1s[args.labeled_bs:].to(device)
            gts = gts[:args.labeled_bs].to(device)

            # EMA prediction
            with torch.no_grad():
                ema_pred_u = ema_model(unlabeled_img)


            pred, e5 = model(torch.cat((unlabeled_img_s, labeled_img)), fp=True)
            pred_u, pred_l = pred.chunk(2)
            e5_u, e5_l = e5.chunk(2)

            pred_d, e5_d = model_depth(torch.cat((unlabeled_depth, labeled_depth)), fp=True)
            pred_u_d, pred_l_d = pred_d.chunk(2)
            e5_u_d, e5_l_d = e5_d.chunk(2)

            # DPA
            unlabeled_img_s_cutmix, ema_pred_u_cutmix = dpa(
                unlabeled_depth1, unlabeled_img_s, ema_pred_u, beta=0.3, t=epoch, T=args.nEpoch
            )
            pred_u_cutmix = model(unlabeled_img_s_cutmix)

            # Losses
            loss_supervised_rgb = criterion(pred_l, gts, threshold=-1)
            loss_supervised_depth = criterion(pred_l_d, gts, threshold=-1)

            loss_u_rgb = criterion(pred_u, ema_pred_u, threshold=args.thresh)
            loss_u_depth = criterion(pred_u_d, ema_pred_u, threshold=args.thresh)

            if args.ud:
                pred_u_up = update_pseudo_labels(ema_pred_u, pred_u_d, gamma=args.thresh_depth)
            else:
                pred_u_up=ema_pred_u

            loss_u_rgb_d = criterion_d(pred_u, pred_u_up, threshold=args.thresh_depth)
            loss_u_rgb_dpa = criterion(pred_u_cutmix, ema_pred_u_cutmix, threshold=args.thresh_dpa)

            loss_rgb = (loss_supervised_rgb + 0.5 * loss_u_rgb + loss_u_rgb_dpa + 0.5 * loss_u_rgb_d) / 3.0
            loss_depth = (loss_supervised_depth + 1.2 * loss_u_depth) / 2.0

            loss_cont = cont_loss(pred_u, pred_l, pred_u_d, pred_l_d)
            loss_mse = mse_consistency_loss(pred_u, pred_u_d)
            loss_con_f = L2_loss(e5_u, e5_u_d)
            loss_con = (loss_cont + 0.5 * loss_mse + 0.5 * loss_con_f) / 2.0

            loss_total = (loss_rgb + loss_depth + loss_con) / 3.0

            # Backward
            optimizer.zero_grad()
            optimizer_depth.zero_grad()
            loss_total.backward()
            optimizer.step()
            optimizer_depth.step()

            # Update EMA
            update_ema_variables(model, ema_model, args.ema_decay, global_iter)

            #Step scheduler once per iter
            scheduler.step()
            scheduler_depth.step()

            global_iter += 1

            # Logging
            progress_bar.set_postfix({
                'iter': global_iter,
                'loss_rgb': f'{loss_rgb.item():.5f}',
                'loss_depth': f'{loss_depth.item():.5f}'
            })

        # Step scheduler once per epoch
        #scheduler.step()
        #scheduler_depth.step()

        # Save checkpoint
        if (epoch + 1) % args.ckpt_period == 0:
            pth_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), pth_path)
            print(f"Saved checkpoint: {pth_path}")

    print(f"Training finished")


if __name__ == '__main__':
    args = parse_args()
    train(args)
