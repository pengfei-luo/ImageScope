python imagescope.py --split test --dataset CIRR --run_name cirr\
 --llm_path "Reasoner LLM model path"\
 --mllm_path "Verifier MLLM model path"\
 --img_cap_model_path "Captioner MLLM model path"\
 --eval_mllm_path "Evalutor MLLM model path"\
 --clip_path "Your VLM model path"\
 --dataset_path "Dataset folder path"

# For CIRR subset setting, add --subset in the command
