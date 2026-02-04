#!/bin/bash

python test.py \
    --en omni-rad-closed \
    --infer_mode local \
    --at closed \
    --dt 1000 \
    --ct all \
    --model Qwen2.5-VL-7B-Instruct \
    --em qwen2.5-vl-instruct \
    --cm Qwen2.5-VL-7B-Instruct \
    --gk false

# python test.py \
#     --en slake \
#     --infer_mode local \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em qwen2.5-vl-instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --gk false

python training_free_grpo_v2.py \
    --en omni-rad-closed \
    --at closed \
    --ct all \
    --infer_mode local \
    --model Qwen2.5-VL-7B-Instruct \
    --em Qwen2.5-VL-7B-Instruct \
    --cm Qwen2.5-VL-7B-Instruct \
    --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
    --ak skd \
    --dt 500
echo "omni-rad-closed training session completed."


python test.py \
    --en omni-rad-closed \
    --at closed \
    --dt 1000 \
    --ct all \
    --infer_mode local \
    --model Qwen2.5-VL-7B-Instruct \
    --em qwen2.5-vl-instruct \
    --cm Qwen2.5-VL-7B-Instruct \
    --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
    --ak skd \
    --kb knowledges/omni-rad-closed_tag_kb_qwen2.5-vl-instruct.pkl


# python test.py \
#     --en omni-rad-closed \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --gk false


# python training_free_grpo_v2.py \
#     --en omni-rad-closed \
#     --at closed \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "omni-rad-closed training session completed."


# python test.py \
#     --en omni-rad-closed \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/slake_tag_kb_qwen2.5-vl-instruct.pkl


# python training_free_grpo_v2.py \
#     --en slake \
#     --at closed \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "slake training session completed."


# python test.py \
#     --en slake \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/omni-rad-closed_versions_qwen2.5-vl-instruct_e1/kb_step_499.pkl

# python training_free_grpo_v2.py \
#     --en vqa-rad \
#     --at closed \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 300
# echo "vqa-rad training session completed."


# python test.py \
#     --en vqa-rad \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/vqa-rad_tag_kb_qwen2.5-vl-instruct.pkl


# python test.py \
#     --en pathvqa \
#     --at closed \
#     --dt 300 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --gk false


# python training_free_grpo_v2.py \
#     --en pathvqa \
#     --at closed \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 300
# echo "pathvqa training session completed."


# python test.py \
#     --en pathvqa \
#     --at closed \
#     --dt 300 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/pathvqa_tag_kb_qwen2.5-vl-instruct.pkl