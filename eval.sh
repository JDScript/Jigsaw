# Use env puzzlefusionpp

CATEGORY=everyday
REDUNDANT=0
REMOVAL=2
CONFIG_NAME=$CATEGORY

# Removal and Redundant can not be both greater than 0
if [ $REDUNDANT -gt 0 ] && [ $REMOVAL -gt 0 ]; then
    echo "Redundant and Removal can not be both greater than 0"
    exit 1
fi

if [ $REDUNDANT -gt 0 ]; then
    CONFIG_NAME=$CATEGORY"_redundant_"$REDUNDANT
fi

if [ $REMOVAL -gt 0 ]; then
    CONFIG_NAME=$CATEGORY"_missing_"$REMOVAL
fi

# python eval_matching.py \
#     --cfg experiments/$CONFIG_NAME.yaml

CUDA_VISIBLE_DEVICES=3 python eval_matching.py \
    --cfg experiments/$CONFIG_NAME.yaml