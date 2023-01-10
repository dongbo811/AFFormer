# base
bash dist_train.sh ./configs/AFFormer/AFFormer_base_ade20k.py 4
bash dist_train.sh ./configs/AFFormer/AFFormer_base_cityscapes.py 2

bash tools/dist_test.sh ./configs/AFFormer/AFFormer_base_ade20k.py ./pretained_weight/AFFormer_base_ade20k.pth 8 --eval mIoU
bash tools/dist_test.sh ./configs/AFFormer/AFFormer_base_cityscapes.py ./pretained_weight/AFFormer_base_cityscapes.pth 8 --eval mIoU

# -----------------------------------------------------------------
# small
bash dist_train.sh ./configs/AFFormer/AFFormer_base_ade20k.py 4
bash dist_train.sh ./configs/AFFormer/AFFormer_base_cityscapes.py 2

bash tools/dist_test.sh ./configs/AFFormer/AFFormer_small_ade20k.py ./pretained_weight/AFFormer_small_ade20k.pth 8 --eval mIoU
bash tools/dist_test.sh ./configs/AFFormer/AFFormer_small_cityscapes.py ./pretained_weight/AFFormer_small_cityscapes.pth 8 --eval mIoU


# -----------------------------------------------------------------
# tiny
bash dist_train.sh ./configs/AFFormer/AFFormer_base_ade20k.py 4
bash dist_train.sh ./configs/AFFormer/AFFormer_base_cityscapes.py 2

bash tools/dist_test.sh ./configs/AFFormer/AFFormer_tiny_ade20k.py ./pretained_weight/AFFormer_tiny_ade20k.pth 8 --eval mIoU
bash tools/dist_test.sh ./configs/AFFormer/AFFormer_tiny_cityscapes.py ./pretained_weight/AFFormer_tiny_cityscapes.pth 8 --eval mIoU

