import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch

from data import get_dataset
from metric import LayoutFID, compute_maximum_iou, \
    compute_overlap, compute_alignment


def average(scores):
    return sum(scores) / len(scores)


def print_scores(score_dict):
    for k, v in score_dict.items():
        if k in ['Alignment', 'Overlap']:
            v = [_v * 100 for _v in v]
        if len(v) > 1:
            mean, std = np.mean(v), np.std(v)
            print(f'\t{k}: {mean:.2f} ({std:.2f})')
        else:
            print(f'\t{k}: {v[0]:.2f}')


def process_dolfin_input_backup(input_tensor):
    h = 600
    w = 400
    x = input_tensor

    data_list = []

    for i in range(16):
        tmpt = torch.zeros(1, h, w)
        tmp = x[0][:, 4*i:4*i+4]


        ############# start here ##############
        b0 = round((tmp[0][0].item() + 1) * (w/2.0))
        b0 = max(0, min(w, b0))
        b1 = round((tmp[0][1].item() + 1) * (h/2.0))
        b1 = max(0, min(h, b1))
        b2 = round((2.0*tmp[0][2].item() + tmp[0][0].item() + 1) * (w/2.0))
        b2 = max(0, min(w, b2))
        b3 = round((2.0*tmp[0][3].item() + tmp[0][1].item() + 1) * (h/2.0))
        b3 = max(0, min(h, b3))
        # print(b0, b1, b2, b3, w, h)
        if (b0+b1+b2+b3>5):
            tmpt[:, b1:b3+1, b0:b2+1] = 1
        else:
            break

        typt = tmp[2:4, :]
        tt = -1
        for t in range(4):
            if (typt[0][t] > 0):
                tt = t
                break
        if (tt == -1) and (typt[1][0] > 0):
            tt = 4
        ############## end here ##################

        # # we have correct b0, b1, b2, b3 and type tt here
        # print(f"b0, b1 ({b0}, {b1})")
        # print(f"b2, b3 ({b2}, {b3})")
        # print(f"tt ({tt})")
        # breakpoint()
        # print()

        # compute center
        center_x = (b0 + b2) / 2
        center_y = (b1 + b3) / 2
        bbox_width = (b2 - b0)
        bbox_height = (b3 - b1)

        # normalize
        center_x = center_x / w
        center_y = center_y / h
        bbox_width = bbox_width / w
        bbox_height = bbox_height / h

        if bbox_width < 0 or bbox_height < 0:
            print(f"error width {bbox_width} bbox {bbox_height}")
            continue

        bbox = torch.tensor([center_x, center_y, bbox_width, bbox_height], dtype=torch.float)
        label = torch.tensor(tt, dtype=torch.long)

        bbox = bbox.unsqueeze(0)
        label = label.unsqueeze(0)

        data = Data(x=bbox, y=label)
        data_list.append(data)

    return data_list

    #     ret_list.append({
    #         "type": tt,
    #         "xywh" : [center_x, center_y, bbox_width, bbox_height]
    #     })

    # return ret_list



def process_dolfin_input(input_tensor):
    h = 600
    w = 400
    x = input_tensor

    data_list = []

    bbox_list = []
    label_list = []

    for i in range(16):
        tmpt = torch.zeros(1, h, w)
        tmp = x[0][:, 4*i:4*i+4]


        ############# start here ##############
        b0 = round((tmp[0][0].item() + 1) * (w/2.0))
        b0 = max(0, min(w, b0))
        b1 = round((tmp[0][1].item() + 1) * (h/2.0))
        b1 = max(0, min(h, b1))
        b2 = round((2.0*tmp[0][2].item() + tmp[0][0].item() + 1) * (w/2.0))
        b2 = max(0, min(w, b2))
        b3 = round((2.0*tmp[0][3].item() + tmp[0][1].item() + 1) * (h/2.0))
        b3 = max(0, min(h, b3))
        # print(b0, b1, b2, b3, w, h)
        if (b0+b1+b2+b3>5):
            tmpt[:, b1:b3+1, b0:b2+1] = 1
        else:
            break

        typt = tmp[2:4, :]
        tt = -1
        for t in range(4):
            if (typt[0][t] > 0):
                tt = t
                break
        if (tt == -1) and (typt[1][0] > 0):
            tt = 4
        ############## end here ##################

        # # we have correct b0, b1, b2, b3 and type tt here
        # print(f"b0, b1 ({b0}, {b1})")
        # print(f"b2, b3 ({b2}, {b3})")
        # print(f"tt ({tt})")
        # breakpoint()
        # print()

        # compute center
        center_x = (b0 + b2) / 2
        center_y = (b1 + b3) / 2
        bbox_width = (b2 - b0)
        bbox_height = (b3 - b1)

        # normalize
        center_x = center_x / w
        center_y = center_y / h
        bbox_width = bbox_width / w
        bbox_height = bbox_height / h

        if bbox_width < 0 or bbox_height < 0:
            print(f"error width {bbox_width} bbox {bbox_height}")
            continue

        bbox = torch.tensor([center_x, center_y, bbox_width, bbox_height], dtype=torch.float)
        label = torch.tensor(tt, dtype=torch.long)

        bbox = bbox.unsqueeze(0)
        label = label.unsqueeze(0)

        bbox_list.append(bbox)
        label_list.append(label)

        

        # data = Data(x=bbox, y=label)
        # data_list.append(data)

    # return data_list

    bbox_tensor = torch.cat(bbox_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)
    # print(bbox_tensor.shape)
    # print(label_tensor.shape)
    # breakpoint()

    # try:
    #     bbox_tensor = torch.cat(bbox_list, dim=0)
    #     label_tensor = torch.cat(label_list, dim=0)
    # except Exception as e:
    #     print(e)
    #     breakpoint()
    #     print()

    # return shape: 16 * 4 for bbox, 16 for label
    return bbox_tensor, label_tensor

    #     ret_list.append({
    #         "type": tt,
    #         "xywh" : [center_x, center_y, bbox_width, bbox_height]
    #     })

    # return ret_list

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dolfin_sample_dir = "/mnt/pentagon/yiw182/DiTC_pbnbb_std/4226sample_sep/DiT-S-4-0132000-size-256-vae-ema-cfg-1.5-seed-0"

    # load evaluation layouts
    args_dataset = "publaynet"

    # we temporary ignore dataloader 
    dataset = get_dataset(args_dataset, 'test')
    test_layouts = [(data.x.numpy(), data.y.numpy()) for data in dataset]

    dolfin_layouts = []

    process_num = 1024
    alignment, overlap = [], []
    for i in tqdm(range(process_num)):
        file_path = os.path.join(dolfin_sample_dir, f"{i}.pt")

        # check if file exist
        assert os.path.exists(file_path)

        dolfin_layout = torch.load(file_path, map_location=torch.device('cpu'))

        bbox_tensor, label_tensor = process_dolfin_input(dolfin_layout)

        bbox_numpy = bbox_tensor.numpy()
        label_numpy = label_tensor.numpy()

        dolfin_layouts.append((
            bbox_numpy, label_numpy
        ))

        bbox_tensor.to(device)
        label_tensor.to(device)

        mask = torch.ones_like(label_tensor).bool()
        # 16 * 4 -> 1 * 16 * 4
        bbox_tensor = bbox_tensor.unsqueeze(0)
        # 16 -> 1 * 16
        mask = mask.unsqueeze(0)

        alignment += compute_alignment(bbox_tensor, mask).tolist()
        overlap += compute_overlap(bbox_tensor, mask).tolist()

    # max_iou = compute_maximum_iou(test_layouts, val_layouts)
    alignment = average(alignment)
    overlap = average(overlap)

    print(f"alignment {alignment}")
    print(f"overlap {overlap}")

    max_iou = compute_maximum_iou(test_layouts, dolfin_layouts)

    print(f"max iou {max_iou}")

    breakpoint()
    print()


# original main start

def main_backup():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', type=str, help='dataset name',
    #                     choices=['rico', 'publaynet', 'magazine'])
    # parser.add_argument('pkl_paths', type=str, nargs='+',
    #                     help='generated pickle path')
    # parser.add_argument('--batch_size', type=int,
    #                     default=64, help='input batch size')
    # parser.add_argument('--compute_real', action='store_true')
    # args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args_dataset = "publaynet"
    args_batch_size = 64
    args_compute_real = True

    # we temporary ignore dataloader 
    dataset = get_dataset(args_dataset, 'test')
    dataloader = DataLoader(dataset,
                            batch_size=args_batch_size,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)
    test_layouts = [(data.x.numpy(), data.y.numpy()) for data in dataset]

    # prepare for evaluation
    fid_test = LayoutFID(args_dataset, device)

    # print("fid test init done")
    # breakpoint()
    # print()

    # real layouts
    alignment, overlap = [], []
    for i, data in tqdm(enumerate(dataloader)):
        data = data.to(device)
        label, mask = to_dense_batch(data.y, data.batch)
        bbox, _ = to_dense_batch(data.x, data.batch)
        padding_mask = ~mask

        fid_test.collect_features(bbox, label, padding_mask,
                                  real=True)

        if args_compute_real:

            # print(bbox.shape)
            # print(mask.shape)
            # # print()
            # breakpoint()

            alignment += compute_alignment(bbox, mask).tolist()
            overlap += compute_overlap(bbox, mask).tolist()

    if args_compute_real:
        dataset = get_dataset(args_dataset, 'val')
        dataloader = DataLoader(dataset,
                                batch_size=args_batch_size,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False)
        val_layouts = [(data.x.numpy(), data.y.numpy()) for data in dataset]

        print(type(val_layouts[0]))
        print(len(val_layouts[0]))
        breakpoint()

        max_iou = compute_maximum_iou(test_layouts, val_layouts)

        for i, data in tqdm(enumerate(dataloader)):
            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask

            fid_test.collect_features(bbox, label, padding_mask)

        fid_score = fid_test.compute_score()

        # breakpoint()

        max_iou = compute_maximum_iou(test_layouts, val_layouts)
        alignment = average(alignment)
        overlap = average(overlap)

        print('Real data:')
        print_scores({
            'FID': [fid_score],
            'Max. IoU': [max_iou],
            'Alignment': [alignment],
            'Overlap': [overlap],
        })
        print()

    print("compute real done")
    breakpoint()
    print()

    # # generated layouts
    # scores = defaultdict(list)
    # for pkl_path in args.pkl_paths:
    #     alignment, overlap = [], []
    #     with Path(pkl_path).open('rb') as fb:
    #         generated_layouts = pickle.load(fb)

    #     for i in range(0, len(generated_layouts), args.batch_size):
    #         i_end = min(i + args.batch_size, len(generated_layouts))

    #         # get batch from data list
    #         data_list = []
    #         for b, l in generated_layouts[i:i_end]:
    #             bbox = torch.tensor(b, dtype=torch.float)
    #             label = torch.tensor(l, dtype=torch.long)
    #             data = Data(x=bbox, y=label)
    #             data_list.append(data)
    #         data = Batch.from_data_list(data_list)

    #         data = data.to(device)
    #         label, mask = to_dense_batch(data.y, data.batch)
    #         bbox, _ = to_dense_batch(data.x, data.batch)
    #         padding_mask = ~mask

    #         fid_test.collect_features(bbox, label, padding_mask)
    #         alignment += compute_alignment(bbox, mask).tolist()
    #         overlap += compute_overlap(bbox, mask).tolist()

    #     fid_score = fid_test.compute_score()
    #     max_iou = compute_maximum_iou(test_layouts, generated_layouts)
    #     alignment = average(alignment)
    #     overlap = average(overlap)

    #     scores['FID'].append(fid_score)
    #     scores['Max. IoU'].append(max_iou)
    #     scores['Alignment'].append(alignment)
    #     scores['Overlap'].append(overlap)

    # print(f'Input size: {len(args.pkl_paths)}')
    # print(f'Dataset: {args.dataset}')
    # print_scores(scores)

# original main end

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', type=str, help='dataset name',
    #                     choices=['rico', 'publaynet', 'magazine'])
    # parser.add_argument('pkl_paths', type=str, nargs='+',
    #                     help='generated pickle path')
    # parser.add_argument('--batch_size', type=int,
    #                     default=64, help='input batch size')
    parser.add_argument('--origin', action='store_true')

    args = parser.parse_args()

    if args.origin:
        main_backup()
    else:
        main()

    # main()
