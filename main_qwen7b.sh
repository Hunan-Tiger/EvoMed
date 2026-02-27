#!/bin/bash

# python test.py \
#     --en omnimed-path \
#     --infer_mode local \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em qwen2.5-vl-instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --gk false

# python test.py \
#     --en vqa-rad \
#     --infer_mode local \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em qwen2.5-vl-instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --gk false

# python training_free_grpo_v2.py \
#     --en omni-rad-closed \
#     --at closed \
#     --ct all \
#     --infer_mode local \
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
#     --infer_mode local \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em qwen2.5-vl-instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/omni-rad-closed_tag_kb_Qwen2.5-VL-7B-Instruct.pkl
    

# SLAKE 3 epoch : Qwen2.5-VL-7B-Instruct
# python training_free_grpo_v2.py \
#     --en omnimed-path \
#     --at closed \
#     --ct all \
#     --infer_mode local \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "omnimed-path training session completed."

# # 重命名knowledges/slake_versions_Qwen2.5-VL-7B-Instruct为knowledges/slake_versions_Qwen2.5-VL-7B-Instruct_e1
# if [ -d "knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct" ]; then
#     mv knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct_e1
#     echo "Renamed knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct to knowledges/omnimed-path_Qwen2.5-VL-7B-Instruct_e1"
# else
#     echo "Directory knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct not found, skipping rename."
# fi

# python test.py \
#     --en omnimed-path \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --infer_mode local \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em qwen2.5-vl-instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct_e1/kb_step_499.pkl


# python training_free_grpo_v2.py \
#     --en omnimed-path \
#     --at closed \
#     --ct all \
#     --infer_mode local \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "omnimed-path training session completed."

# # 重命名knowledges/slake_versions_Qwen2.5-VL-7B-Instruct为knowledges/slake_versions_Qwen2.5-VL-7B-Instruct_e1
# if [ -d "knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct" ]; then
#     mv knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct_e2
#     echo "Renamed knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct to knowledges/omnimed-path_Qwen2.5-VL-7B-Instruct_e2"
# else
#     echo "Directory knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct not found, skipping rename."
# fi

# python test.py \
#     --en omnimed-path \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --infer_mode local \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em qwen2.5-vl-instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/omnimed-path_tag_kb_Qwen2.5-VL-7B-Instruct.pkl


# python training_free_grpo_v2.py \
#     --en omnimed-path \
#     --at closed \
#     --ct all \
#     --infer_mode local \
#     --model Qwen2.5-VL-7B-Instruct \
#     --em Qwen2.5-VL-7B-Instruct \
#     --cm Qwen2.5-VL-7B-Instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --dt 500 \
#     --ak skd \

# echo "omnimed-path training session completed."

# # 重命名knowledges/slake_versions_Qwen2.5-VL-7B-Instruct为knowledges/slake_versions_Qwen2.5-VL-7B-Instruct_e1
# if [ -d "knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct" ]; then
#     mv knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct_e3
#     echo "Renamed knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct to knowledges/omnimed-path_Qwen2.5-VL-7B-Instruct_e3"
# else
#     echo "Directory knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct not found, skipping rename."
# fi


python test.py \
    --en omnimed-path \
    --at closed \
    --dt 1000 \
    --ct all \
    --infer_mode local \
    --model Qwen2.5-VL-7B-Instruct \
    --em qwen2.5-vl-instruct \
    --cm Qwen2.5-VL-7B-Instruct \
    --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
    --ak skd \
    --kb knowledges/omnimed-path_versions_qwen2.5-vl-instruct_e3/kb_step_499.pkl


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
    --kb knowledges/omni-rad-closed_versions_qwen2.5-vl-instruct_e3/kb_step_499.pkl


python test.py \
    --en slake \
    --at closed \
    --dt 1000 \
    --ct all \
    --infer_mode local \
    --model Qwen2.5-VL-7B-Instruct \
    --em qwen2.5-vl-instruct \
    --cm Qwen2.5-VL-7B-Instruct \
    --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
    --ak skd \
    --kb knowledges/slake_versions_qwen2.5-vl-instruct_e3/kb_step_499.pkl
