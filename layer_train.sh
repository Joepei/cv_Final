#!/bin/bash

content_dir="examples/content/in"
style_dir='examples/style/tar'
content_seg_dir="examples/content_segment/in"
style_seg_dir='examples/style_segment/tar'
declare -a layers=(4)
iteration_num='18490'
decoder="decoder_10epoch/dec_1_${iteration_num}.pkl,decoder_10epoch/dec_2_${iteration_num}.pkl,decoder_10epoch/dec_3_${iteration_num}.pkl,decoder_10epoch/dec_4_${iteration_num}.pkl"
smooth="gif"
encoder="2"
seg=true

for i in {1..1}; do
    for layer in ${layers[@]}; do
        if [ $i -eq 23 ]; then
            continue
        fi
        echo "${content_dir}${i}.png, ${style_dir}${i}.png, layers: $layer"
        if [ "$seg" = true ]; then
            # echo "python3 run_wct_segmentation.py --x $layer --style ${style_dir}${i}.png --content ${content_dir}${i}.png --output outputs_automated/out${i} --decoder $decoder"
            python3 run_wct_segmentation.py --x "$layer" --style "${style_dir}${i}.png" \
                                            --content "${content_dir}${i}.png" \
                                            --style_seg "${style_seg_dir}${i}.png" \
                                            --content_seg "${content_seg_dir}${i}.png" \
                                            --output "outputs_automated/out${i}_seg_l${layer}" \
                                            --decoder "${decoder}" --smooth "${smooth}" --encoder "${encoder}"
        else 
            python3 run_wct_segmentation.py --x "$layer" --style "${style_dir}${i}.png" \
                                            --content "${content_dir}${i}.png" \
                                            --output "outputs_automated/out${i}_noseg_l${layer}" \
                                            --decoder "${decoder}" --smooth "${smooth}" \
                                            --encoder "${encoder}"
        fi
    done
done
