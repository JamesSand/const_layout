
export CUDA_VISIBLE_DEVICES=2

# # for unique docsim metric 
# # test set(priority1) /mnt/pentagon/yiw182/DiTC_pbnbb_std/testset/tensor_test
# # content1 /mnt/pentagon/yiw182/DiTC_pbnbb_std/4226sample_sep/DiT-S-4-0132000-size-256-vae-ema-cfg-1.5-seed-0
# # content2 /mnt/pentagon/yiw182/DiTC_pbnbb_std/4226sample_org/DiT-S-4-0276000-size-256-vae-ema-cfg-1.5-seed-0
# python eval.py


# for layoutvae
python eval2.py


