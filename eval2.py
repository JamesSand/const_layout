import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
from tqdm import tqdm

from multiprocessing import Pool, cpu_count


# import numpy as np
from scipy.optimize import linear_sum_assignment

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



def process_dolfin_input(input_tensor):
    h = 600
    w = 400
    x = input_tensor

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

    bbox_tensor = torch.cat(bbox_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)

    # return shape: 16 * 4 for bbox, 16 for label
    return bbox_tensor, label_tensor


def process_publaynet_gt(process_num):
    # process_num = 1024
    publaynet_gt_dir = "/mnt/pentagon/yiw182/DiTC_pbnbb_std/testset/tensor_test"

    test_layouts = []
    for i in tqdm(range(process_num), desc="load publaynet gt"):
        file_path = os.path.join(publaynet_gt_dir, f"test_{i}.pt")
        layout_tensor = torch.load(file_path, map_location=torch.device('cpu'))
        bbox_tensor = layout_tensor[:, :4]
        label_tensor = layout_tensor[:, 4]

        bbox_numpy = bbox_tensor.numpy()
        label_numpy = label_tensor.numpy()

        test_layouts.append((
            bbox_numpy, label_numpy
        ))

    return test_layouts


def docsim_bbox_weight(bbox1, bbox2):
    # and suppose they have same type(label)
    # suppose input format is (center_x, center_y, width, height)
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2

    alpha = min(w1 * h1, w2 * h2) ** 0.5

    exponent = - ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 - 2 * (abs(w1 - w2) + abs(h1 - h2))
    exp = 2 ** exponent

    return alpha * exp

def docsim_layout_weight(layout1, layout2):

    bboxes1, labels1 = layout1
    bboxes2, labels2 = layout2

    bbox_num1 = bboxes1.shape[0]
    bbox_num2 = bboxes2.shape[0]
    
    bbox_weight_matrix = np.full((bbox_num1, bbox_num2), 0.0)

    for i in range(bbox_num1):
        # # change all negative value into 0.0
        # bbox1 = bboxes1[i]
        # bbox1[bbox1 < 0] = 0.0
        # bboxes1[i] = bbox1
        for j in range(bbox_num2):
            if labels1[i] != labels2[j]:
                continue
            bbox_weight_matrix[i][j] = docsim_bbox_weight(bboxes1[i], bboxes2[j])

            # check if nan
            if np.isnan(bbox_weight_matrix[i][j]):
                print("is nan")
                print(bboxes1[i])
                print(bboxes2[j])
                breakpoint()
                print()

    try:
        if bbox_weight_matrix.max() == 0.0:
            return 0.0
    except Exception as e:
        print(e)
        print(bbox_weight_matrix)
        breakpoint()
        print()

    # # use hungarian matching to get the final score
    # row_ind, col_ind = linear_sum_assignment(- bbox_weight_matrix) 

    try:
        row_ind, col_ind = linear_sum_assignment(- bbox_weight_matrix) 
    except Exception as e:
        print(e)
        print(bbox_weight_matrix)
        breakpoint()
        print()

    total = 0.0
    for i, j in zip(row_ind, col_ind):
        total += bbox_weight_matrix[i][j]

    # we use no average here
    return total

# calculate uni match count
def unique_match(selected_metric, threshold):
    unique_match_cnt = 0

    for select_item in selected_metric:
        if select_item >= threshold:
            unique_match_cnt += 1

    return unique_match_cnt


def calculate_uni_match_docsim(layouts1, layouts2):

    ############ multi process version ###########
    num_processes = cpu_count()  # 获取CPU核心数
    pool = Pool(processes=num_processes)
    row_result_list = []
    for i in tqdm(range(len(layouts1)), desc="calculate docsim"):
        # 分配任务
        row_result = [pool.apply_async(docsim_layout_weight, args=(layouts1[i], layouts2[j])) for j in range(len(layouts2))]

        # 收集结果
        row_result = [p.get() for p in row_result]
        row_result_list.append(row_result)

    layout_weight_matrix = np.stack(row_result_list, axis=0)
    ############ multi process version ###########


    # ############ normal version ###########
    # layout_weight_matrix = np.full((len(layouts1), len(layouts2)), -np.inf)
    # for i in tqdm(range(len(layouts1)), desc="calculate docsim"):
    #     for j in range(len(layouts2)):
    #         layout_weight_matrix[i][j] = docsim_layout_weight(layouts1[i], layouts2[j])
    # ############ normal version ###########


    # use hungarian matching to get the final score
    row_ind, col_ind = linear_sum_assignment(- layout_weight_matrix) 

    selected_weight = []
    for i, j in zip(row_ind, col_ind):
        selected_weight.append(layout_weight_matrix[i][j])

    # we set threshold to 0.5
    return unique_match(selected_weight, 0.5)

########### layout vae start ############
def process_layoutvae_input(bbox, label):

    # 16 * 1 -> 16
    label = label[:, 0]

    # converting label from 1-5 to 0-4
    label = label - 1
    # save only geq labels
    positive_elements = label >= 0

    # 找到所有大于零的元素的索引
    indices = np.where(positive_elements)

    bbox = bbox[indices]

    # change bbox negative value into zero
    bbox[bbox < 0] = 0.0

    label = label[indices]

    return bbox, label

def get_layoutvae_input():
    numpy_bbox_path = "/mnt/pentagon/yiw182/layout-generation/LayoutVAE/boxes.npy"
    numpy_label_path = "/mnt/pentagon/yiw182/layout-generation/LayoutVAE/cates.npy"

    bbox_numpy = np.load(numpy_bbox_path)
    label_numpy = np.load(numpy_label_path)

    layoutvae_layouts = []
    for i in range(bbox_numpy.shape[0]):
        bbox_np, label_np = process_layoutvae_input(bbox_numpy[i], label_numpy[i])

        layoutvae_layouts.append((
            bbox_np, label_np
        ))

    return layoutvae_layouts
########### layout vae end ############

########## lgdata start ############
def process_lgdata_input(layout_numpy):
    # input shape should be (9, 5)
    bbox = layout_numpy[:, :4]
    label = layout_numpy[:, 4]

    bbox[bbox < 0] = 0.0

    return bbox, label

def get_lgdata_input():
    file_path = "/mnt/pentagon/yiw182/layout-generation/LayoutGAN/lgdata771.npy"

    print("*" * 50)
    print(file_path)
    print("*" * 50)

    layout_np = np.load(file_path)

    lgdata_layouts = []
    for i in range(layout_np.shape[0]):
        bbox_np, label_np = process_lgdata_input(layout_np[i])
        lgdata_layouts.append((
            bbox_np, label_np
        ))

    return lgdata_layouts
######### lgdata end ###########

############ pbn start #########
def process_pbn_input(bbox, label):
    # process label first
    label = label - 1
    # save only geq labels
    positive_elements = label >= 0

    # 找到所有大于零的元素的索引
    indices = np.where(positive_elements)

    bbox = bbox[indices]
    label = label[indices]

    # change bbox negative value into zero
    bbox[bbox < 0] = 0.0
    # the input of bbox is left x and low y
    for i in range(bbox.shape[0]):
        lowx, lowy, w, h = bbox[i]
        centerx = lowx + w / 2
        centery = lowy + h / 2
        bbox[i][0] = centerx
        bbox[i][1] = centery

    return bbox, label

def get_pbn_input():
    bbox_path = "/mnt/pentagon/yiw182/const_layout/pbn_b1000.npy"
    label_path = "/mnt/pentagon/yiw182/const_layout/pbn_c1000.npy"

    print("*" * 50)
    print("bbox path")
    print(bbox_path)
    print("label path")
    print(label_path)
    print("*" * 50)

    bbox_np = np.load(bbox_path)
    label_np = np.load(label_path)

    pbn_layouts = []
    for i in tqdm(range(bbox_np.shape[0])):
        bbox, label = process_pbn_input(bbox_np[i], label_np[i])

        pbn_layouts.append((
            bbox, label
        ))

    return pbn_layouts

    # print(bbox_np.shape)
    # print(label_np.shape)
    # breakpoint()
    # print()
############ pbn end #########


def process_rico_gt(process_num):
    # process num should be 1024
    # and we take last 1024 sample
    rico_gt_path = "/mnt/pentagon/yiw182/szz/yilin_data/test_ts"
    test_layouts = []

    # empty_cnt = 0

    # total_num = 4225

    cur_ind = 4225

    # for i in tqdm(range(process_num), desc="loading rico gt"):

    with tqdm(total=process_num, desc="loading rico gt") as pbar:
            
        while True:
            # cur_ind = total_num - i
            file_path = os.path.join(rico_gt_path, f"test_{cur_ind}.pt")

            cur_ind -= 1

            layout_tensor = torch.load(file_path, map_location=torch.device("cpu"))

            bbox_tensor = layout_tensor[:, :4]
            label_tensor = layout_tensor[:, 4]

            # avoid negative value
            bbox_tensor[bbox_tensor < 0] = 0.0

            bbox_np = bbox_tensor.numpy()
            label_np = label_tensor.numpy()

            # print(bbox_np.shape)
            # print(label_np.shape)
            # breakpoint()

            if bbox_np.shape[0] == 0:
                continue
                # print(file_path)
                # print(bbox_np)
                # print(label_np)
                # # breakpoint()
                # # print()

                # empty_cnt += 1

            pbar.update(1)

            test_layouts.append((
                bbox_np, label_np
            ))

            if len(test_layouts) == process_num:
                break

    # print(empty_cnt)
    # breakpoint()
    # print()
    
    return test_layouts

def get_sample_sep_ts_input():
    file_dir = "/mnt/pentagon/yiw182/szz/yilin_data/sample_sep_ts"

    print("*" * 50)
    print(file_dir)
    print("*" * 50)

    sample_sep_ts_layouts = []
    total_num = 1024

    # empty_cnt = 0

    for i in tqdm(range(total_num), desc="load sample sep ts"):
        file_path = os.path.join(file_dir, f"test_{i}.pt")
        layout_tensor = torch.load(file_path, map_location=torch.device("cpu"))

        bbox_tensor = layout_tensor[:, :4]
        label_tensor = layout_tensor[:, 4]

        # avoid negative value
        bbox_tensor[bbox_tensor < 0] = 0.0

        bbox_np = bbox_tensor.numpy()
        label_np = label_tensor.numpy()

        if bbox_np.shape[0] == 0:
            continue
            # print(file_path)
            # print(bbox_np)
            # print(label_np)
            # empty_cnt += 1
            # breakpoint()
            # print()

        sample_sep_ts_layouts.append((
            bbox_np, label_np
        ))

    # print(empty_cnt)
    # breakpoint()
    # print()

    return sample_sep_ts_layouts



def get_ts5_input():
    origin_file_path = "/scratch/yiw182/rico_sample/sample_std/ts5"
    file_dir = "/mnt/pentagon/yiw182/szz/yilin_data/ts5"

    print("*" * 50)
    print("origin file path")
    print(origin_file_path)
    print("cur file path")
    print(file_dir)
    print("*" * 50)

    sample_sep_ts_layouts = []
    total_num = 1024

    for i in tqdm(range(total_num), desc="load ts5"):
        file_path = os.path.join(file_dir, f"test_{i}.pt")
        layout_tensor = torch.load(file_path, map_location=torch.device("cpu"))

        bbox_tensor = layout_tensor[:, :4]
        label_tensor = layout_tensor[:, 4]

        # avoid negative value
        bbox_tensor[bbox_tensor < 0] = 0.0

        bbox_np = bbox_tensor.numpy()
        label_np = label_tensor.numpy()

        if bbox_np.shape[0] == 0:
            continue

        sample_sep_ts_layouts.append((
            bbox_np, label_np
        ))

    return sample_sep_ts_layouts




def main():

    ############# getpublaynet test data start ###########
    # processed_layouts = get_layoutvae_input()
    
    # processed_layouts = get_lgdata_input()

    # processed_layouts = get_pbn_input()
    ############ get publaynet test data end ############

    ############ get rico test data start ############

    # processed_layouts = get_sample_sep_ts_input()
    processed_layouts = get_ts5_input()

    ############ get rico test data end ############


    ############ get gt data start ###########
    process_num = 1024

    # test_layouts = process_publaynet_gt(process_num=process_num)

    test_layouts = process_rico_gt(process_num=process_num)

    # print("rico gt load success")
    # breakpoint()
    # print()

    ########### get gt data end ##############

    uni_match_docsim = calculate_uni_match_docsim(processed_layouts, test_layouts)

    print("uni match docsim", uni_match_docsim)
    # breakpoint()
    # print()

    # if you only need unique match of docsim, you should stop here

    # the following is calculate alignment score for processed layouts
    alignment = []
    for bbox_np, label_np in processed_layouts:
        bbox_tensor = torch.from_numpy(bbox_np)
        label_tensor = torch.from_numpy(label_np)

        mask = torch.ones_like(label_tensor).bool()
        bbox_tensor = bbox_tensor.unsqueeze(0)
        mask = mask.unsqueeze(0)

        alignment += compute_alignment(bbox_tensor, mask).tolist()

    alignment = average(alignment)
    alignment *= 100

    print(f"alignment {alignment}")
    breakpoint()
    print()

    dolfin_layouts = []

    process_num = 1024
    alignment, overlap = [], []
    for i in tqdm(range(process_num), desc="dolfin"):
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
        # mask = torch.zeros_like(label_tensor).bool()
        # 16 * 4 -> 1 * 16 * 4
        bbox_tensor = bbox_tensor.unsqueeze(0)
        # 16 -> 1 * 16
        mask = mask.unsqueeze(0)

        if debug_mode:
            continue

        alignment += compute_alignment(bbox_tensor, mask).tolist()
        overlap += compute_overlap(bbox_tensor, mask).tolist()

    # max_iou = compute_maximum_iou(test_layouts, val_layouts)

    # max_iou = compute_maximum_iou(test_layouts, dolfin_layouts)
    # print(f"max iou {max_iou}")
    # breakpoint()

    alignment = average(alignment)
    overlap = average(overlap)

    # multiply 100 to alignment and overlap
    alignment *= 100
    overlap *= 100

    print(f"alignment {alignment}")
    print(f"overlap {overlap}")

    

    breakpoint()
    print()


main()
