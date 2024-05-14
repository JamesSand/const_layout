
import torch
import os

publaynet_gt_dir = "/mnt/pentagon/yiw182/DiTC_pbnbb_std/testset/tensor_test"

for i in range(10):
    file_path = os.path.join(publaynet_gt_dir, f"test_{i}.pt")

# file_path = "/mnt/pentagon/yiw182/DiTC_pbnbb_std/testset/tensor_test/test_0.pt"
    layout_tensor = torch.load(file_path)
    print("*" * 50)
    print(i)
    print(layout_tensor.shape)
    breakpoint()
    print()

