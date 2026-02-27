#!/bin/bash

# python test.py \
#     --en omni-rad-closed \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --gk false

# python test.py \
#     --en pathvqa \
#     --at closed \
#     --dt 300 \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --gk false

# python training_free_grpo_v2.py \
#     --en omnimed-path \
#     --at closed \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "omni-rad-closed training session completed."

# if [ -d "knowledges/omnimed-path_versions_qwen2.5-vl-instruct" ]; then
#     mv knowledges/omnimed-path_versions_qwen2.5-vl-instruct knowledges/omnimed-path_versions_qwen2.5-vl-instruct_e2
#     echo "Renamed knowledges/omnimed-path_versions_qwen2.5-vl-instruct to knowledges/omnimed-path_qwen2.5-vl-instruct_e2"
# else
#     echo "Directory knowledges/omnimed-path_versions_qwen2.5-vl-instruct not found, skipping rename."
# fi

# python test.py \
#     --en omnimed-path \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/omnimed-path_tag_kb_qwen2.5-vl-instruct.pkl

# python training_free_grpo_v2.py \
#     --en omnimed-path \
#     --at closed \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "omni-rad-closed training session completed."

# if [ -d "knowledges/omnimed-path_versions_qwen2.5-vl-instruct" ]; then
#     mv knowledges/omnimed-path_versions_qwen2.5-vl-instruct "$path_"
#     echo "Renamed knowledges/omnimed-path_versions_qwen2.5-vl-instruct to $path_"
# else
#     echo "Directory knowledges/omnimed-path_versions_qwen2.5-vl-instruct not found, skipping rename."
# fi


python test.py \
    --en slake \
    --at closed \
    --dt 1000 \
    --ct all \
    --model qwen2.5-vl-instruct \
    --em qwen2.5-vl-instruct \
    --cm qwen2.5-vl-instruct \
    --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
    --ak skd \
    --kb knowledges/slake_versions_Qwen2.5-VL-7B-Instruct_e3/kb_step_499.pkl


python test.py \
    --en omni-rad-closed \
    --at closed \
    --dt 1000 \
    --ct all \
    --model qwen2.5-vl-instruct \
    --em qwen2.5-vl-instruct \
    --cm qwen2.5-vl-instruct \
    --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
    --ak skd \
    --kb knowledges/omni-rad-closed_versions_Qwen2.5-VL-7B-Instruct_e3/kb_step_499.pkl

python test.py \
    --en omnimed-path \
    --at closed \
    --dt 1000 \
    --ct all \
    --model qwen2.5-vl-instruct \
    --em qwen2.5-vl-instruct \
    --cm qwen2.5-vl-instruct \
    --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
    --ak skd \
    --kb knowledges/omnimed-path_versions_Qwen2.5-VL-7B-Instruct_e3/kb_step_499.pkl


# python training_free_grpo_v2.py \
#     --en omni-rad-closed \
#     --at closed \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "omni-rad-closed training session completed."


# python test.py \
#     --en omni-rad-closed \
#     --at closed \
#     --dt 1000 \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/slake_tag_kb_qwen2.5-vl-instruct.pkl


# python training_free_grpo_v2.py \
#     --en pathvqa \
#     --at closed \
#     --ct all \
#     --reranker_threshold 0.9 \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --dt 500
# echo "pathvqa training session completed."

# if [ -d "knowledges/pathvqa_versions_qwen2.5-vl-instruct" ]; then
#     mv knowledges/pathvqa_versions_qwen2.5-vl-instruct knowledges/pathvqa_versions_qwen2.5-vl-instruct_e1
#     echo "Renamed knowledges/pathvqa_versions_qwen2.5-vl-instruct to knowledges/pathvqa_versions_qwen2.5-vl-instruct_e1"
# else
#     echo "Directory knowledges/pathvqa_versions_qwen2.5-vl-instruct not found, skipping rename."
# fi




# python test.py \
#     --en vqa-rad \
#     --at closed \
#     --dt 1000 \
#     --ts 0.9 \
#     --ct all \
#     --model qwen2.5-vl-instruct \
#     --em qwen2.5-vl-instruct \
#     --cm qwen2.5-vl-instruct \
#     --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start \
#     --ak skd \
#     --kb knowledges/slake_versions_qwen2.5-vl-instruct_e3/kb_step_499.pkl