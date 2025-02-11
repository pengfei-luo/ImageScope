import torch
import copy
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import CLIPModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from utils.data import ImageFeatureDataset
from utils.function import (
    resize_and_concatenate, 
    concatenate_images_with_reference,
    resize_image_ratio,
)
import ray
import os
import time
import random
from copy import deepcopy

@ray.remote
def llm_load_and_inference_ray(llm_path, input_prompt):
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
    model = LLM(
        model=llm_path,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
    )
    tokenizer = model.get_tokenizer()
    _sta_time = time.time()
    conv_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in input_prompt
    ]
    llm_outputs = model.generate(conv_prompts, sampling_params)
    _end_time = time.time()
    _time_interval = _end_time - _sta_time
    info_dict = {
        "time": _time_interval,
        "num_task": len(input_prompt)
    }
    return (llm_outputs, info_dict)


@ray.remote
def mllm_load_and_inference_ray(mllm_path, mllm_query):
    mllm_outputs = defaultdict(list)
    mini_batch_size = 16000
    stop_tokens = None
    max_model_len = None
    if 'llava' in mllm_path:
        prompt = "USER: <image>\n{}\nASSISTANT:"
    elif 'paligemma' in mllm_path:
        prompt = '{}'
    elif 'Phi-3' in mllm_path:
        prompt = "<|user|>\n<|image_1|>\n{}<|end|>\n<|assistant|>\n"
    elif 'InternVL2' in mllm_path:
        prompt = [{
            'role': 'user', 
            'content': "<image>\n{}"
            }]
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        max_model_len = 8192
    elif 'MiniCPM' in mllm_path:
        prompt = [{'role': 'user', 'content': '(<image>./</image>)\n{}'}]
        max_model_len = 4096
        
    if 'Phi-3' in mllm_path:
        model = LLM(
            model=mllm_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            max_num_seqs=5,
        )
    else:
        model = LLM(
            model=mllm_path,
            enforce_eager=True,
            gpu_memory_utilization=0.95,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
    tokenizer = model.get_tokenizer()
    stop_token_ids = None if stop_tokens is None else [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    if 'MiniCPM' in mllm_path: stop_token_ids = [tokenizer.eos_id]
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=1, 
        max_tokens=2048,
        stop_token_ids=stop_token_ids,
    )   

    mllm_inputs = defaultdict(list)
    item_idx_list = list(mllm_query.keys())
    total_num = len(item_idx_list)
    current_query_num = 0
    processed_num = 0
    _num_input_token = []
    _num_output_token = []

    _sta_time = time.time()
    for item_idx in item_idx_list:
        text_inputs = mllm_query[item_idx]['text_inputs']
        if mllm_query[item_idx]['ref_img_path'] is None:
            ref_image_obj = Image.new('RGB', (224, 224))
        else:
            ref_image_obj = Image.open(mllm_query[item_idx]['ref_img_path']).convert('RGB')
        

        top_k_ranked_candidates = mllm_query[item_idx]['top_k_ranked_candidates']
        for candidate_idx in top_k_ranked_candidates.keys():
            candidate_image_path = top_k_ranked_candidates[candidate_idx]['image_path']
            candidate_image_obj = Image.open(candidate_image_path).convert('RGB')
            
            if 'InternVL2' in mllm_path or 'MiniCPM' in mllm_path:
                candidate_image_obj = candidate_image_obj.resize((400, 400))
            # combined_image_obj = concatenate_images_with_reference([ref_image_obj, candidate_image_obj])
            
            for proposition_idx, text_input in enumerate(text_inputs):
                if isinstance(prompt, str):
                    item_prompt = prompt.format(text_input)
                elif isinstance(prompt, list):
                    message = deepcopy(prompt)
                    message[0]['content'] = prompt[0]['content'].format(text_input)
                    item_prompt = tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                mllm_inputs['index'].append((item_idx, candidate_idx, proposition_idx))
                mllm_inputs['inputs'].append({
                    "prompt": item_prompt, 
                    "multi_modal_data": {"image": candidate_image_obj}
                })

        current_query_num += 1
        if len(mllm_inputs['index']) >= mini_batch_size or item_idx == item_idx_list[-1]:
            print('---MLLM Batch Inference from {} to {} , total {} ---'.format(processed_num, 
                                                                            processed_num + current_query_num,
                                                                            total_num))
            processed_num += current_query_num
            current_query_num = 0
            
            outputs = model.generate(mllm_inputs['inputs'], sampling_params)
            for (item_idx, candidate_idx, proposition_idx), o in zip(mllm_inputs['index'], outputs):
                generated_text = o.outputs[0].text
                _num_input_token.append(len(o.prompt_token_ids))
                _num_output_token.append(len(o.outputs[0].token_ids))
                if item_idx not in mllm_outputs.keys():
                    mllm_outputs[item_idx] = defaultdict(list)
                mllm_outputs[item_idx][candidate_idx].append(generated_text)
            mllm_inputs.clear()

    _end_time = time.time()
    _time_interval = _end_time - _sta_time
    info_dict = {
        'time': _time_interval, 
        'num_task': len(_num_input_token), 
        'num_input_token': _num_input_token,
        'num_output_token': _num_output_token
    }

    return (mllm_outputs, info_dict)


@ray.remote
def stage3_large_mllm_inference_qa_ray(mllm_path, mllm_query):
    mllm_outputs = defaultdict(list)
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=1, 
        max_tokens=256,
    )
    model = LLM(
        model=mllm_path,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
    )
    if 'llava-v1.6' in mllm_path:
       prompt = "[INST] <image>\n{} [/INST]"
    else:
        raise NotImplementedError
    
    _total_time = 0
    _total_task = 0
    _keys = list(mllm_query.keys())
    mllm_query_flag = {k: False for k in mllm_query.keys()}
    info_dict = dict()

    # Perpare mllm_turn_query for a single turn
    for turn_idx in range(len(mllm_query[_keys[0]]['top_candidate'])):
        print(f"Turn {turn_idx}")
        _sta_time = time.time()
        mllm_turn_inputs = {"index": list(), "inputs": list()}
        for item_idx in mllm_query.keys():
            if mllm_query_flag[item_idx]:
                continue
            reference_img_path = mllm_query[item_idx]['ref_img_path']
            if reference_img_path is None:
                reference_img_obj = Image.new('RGB', (224, 224))
            else:
                reference_img_obj = Image.open(reference_img_path).convert('RGB')
            candidate_img_path = mllm_query[item_idx]['top_candidate'][turn_idx]['image_path']
            candidate_img_obj = Image.open(candidate_img_path).convert('RGB')
            text_input = mllm_query[item_idx]['text_input']
            
            combined_img_obj = resize_and_concatenate(reference_img_obj, candidate_img_obj)

            mllm_turn_inputs['index'].append((item_idx, turn_idx))
            mllm_turn_inputs['inputs'].append({
                "prompt": prompt.format(text_input), 
                "multi_modal_data": {"image": candidate_img_obj}
            })
        _total_task += len(mllm_turn_inputs['index'])

        # Request MLLM to get response
        outputs = model.generate(mllm_turn_inputs['inputs'], sampling_params)

        # Parse the response
        for (item_idx, turn_idx), o in zip(mllm_turn_inputs['index'], outputs):
            generated_text = o.outputs[0].text
            decision = False
            if generated_text.lower().strip().startswith('yes'):
                mllm_query_flag[item_idx] = True
                decision = True
            
            if item_idx <= 10:
                print(f"{item_idx}, {turn_idx}\n {generated_text}")
            
            mllm_outputs[item_idx].append({
                'turn': turn_idx,
                'decision': decision,
                'text': generated_text,
                'num_input_token': len(o.prompt_token_ids),
                'num_output_token': len(o.outputs[0].token_ids),
            })

        _end_time = time.time()
        _turn_time_interval = _end_time - _sta_time
        _total_time += _turn_time_interval
        info_dict[f"turn_{turn_idx}_time"] = _turn_time_interval

        if all(mllm_query_flag.values()):
            break
    
    info_dict = {
        'time': _total_time, 
        'num_task': _total_task
    }
    return (mllm_outputs, info_dict)


@ray.remote
def stage3_large_mllm_inference_batch_cmp_ray(mllm_path, mllm_query):
    mllm_outputs = defaultdict(list)
    
    stop_tokens = None
    other_args = {}
    if 'llava-v1.6-mistral' in mllm_path:
       prompt = "[INST] <image>\n{} [/INST]"
    elif 'llava-v1.6-vicuna' in mllm_path:
       prompt = "USER: <image>\n{}\nASSISTANT:"
    elif 'llava-v1.6-34b' in mllm_path:
       prompt = "<|im_start|>user\n<image>\n{}<|im_end|>\n"
       other_args = {
            "quantization": "fp8",
            "trust_remote_code": True
        }
    elif 'InternVL2' in mllm_path:
        prompt = [{'role': 'user', 'content': "<image>\n{}"}]
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        other_args = {
            "trust_remote_code": True
        }
    else:
        raise NotImplementedError
    
    model = LLM(
        model=mllm_path,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        max_model_len=8192,
        **other_args
    )
    tokenizer = model.get_tokenizer()
    stop_token_ids = None if stop_tokens is None else [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=1, 
        max_tokens=4096,
        stop_token_ids=stop_token_ids,
        repetition_penalty=1
    )
    
    _total_time = 0
    _total_task = 0
    _keys = list(mllm_query.keys())
    mllm_query_flag = {k: False for k in mllm_query.keys()}
    info_dict = dict()
    info_dict['turn_time'] = list()
    info_dict['turn_task'] = list()

    mllm_outputs = defaultdict(list)
    mllm_inputs = {"index": list(), "inputs": list()}
    _sta_time = time.time()
        
    # Perpare mllm_turn_query for a single turn
    for turn_idx in range(len(mllm_query[_keys[0]]['top_candidate'])):
        print(f"Turn {turn_idx}")
        _sta_time = time.time()
        mllm_turn_inputs = {"index": list(), "inputs": list()}
        for item_idx in mllm_query.keys():
            if mllm_query_flag[item_idx]:
                continue
            
            reference_img_path = mllm_query[item_idx]['ref_img_path']
            if mllm_query[item_idx]['ref_img_path'] is not None:
                reference_img_obj = Image.open(reference_img_path).convert('RGB').resize((800, 800))
            else:
                reference_img_obj = Image.new('RGB', (800, 800))
            candidate_img_path = mllm_query[item_idx]['top_candidate'][turn_idx]['image_path']
            candidate_img_obj = Image.open(candidate_img_path).convert('RGB').resize((800, 800))
            text_input = mllm_query[item_idx]['text_input']
            
            combined_img_obj = concatenate_images_with_reference([reference_img_obj, candidate_img_obj])

            mllm_turn_inputs['index'].append((item_idx, turn_idx))
            message = deepcopy(prompt)
            message[0]['content'] = prompt[0]['content'].format(text_input)
            mllm_input = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            mllm_turn_inputs['inputs'].append({
                "prompt": mllm_input,
                "multi_modal_data": {"image": combined_img_obj}
            })
        _total_task += len(mllm_turn_inputs['index'])

        # Request MLLM to get response
        outputs = model.generate(mllm_turn_inputs['inputs'], sampling_params)

        # Parse the response
        for (item_idx, turn_idx), o in zip(mllm_turn_inputs['index'], outputs):
            generated_text = o.outputs[0].text
            decision = False
            if "ANSWER: Yes" in generated_text:
                mllm_query_flag[item_idx] = True
                decision = True
            
            mllm_outputs[item_idx].append({
                'turn': turn_idx,
                'decision': decision,
                'text': generated_text,
                'num_input_token': len(o.prompt_token_ids),
                'num_output_token': len(o.outputs[0].token_ids),
            })

        _end_time = time.time()
        _turn_time_interval = _end_time - _sta_time
        _total_time += _turn_time_interval
        info_dict["turn_time"].append(_turn_time_interval)
        info_dict["turn_task"].append(len(mllm_turn_inputs['index']))
        

        if all(mllm_query_flag.values()):
            break
    
    info_dict['time'] = _total_time
    info_dict['num_task'] = _total_task
    return (mllm_outputs, info_dict)


@ray.remote
def mllm_inference_for_caption_ray(mllm_path, mllm_query):
    mllm_outputs = defaultdict(list)
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=1, 
        max_tokens=256,
        repetition_penalty=1.05
    )
    model = LLM(
        model=mllm_path,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
    )
    prompt = "USER:<image>\n{}\nASSISTANT:"
    mini_batch_size = 10000
    keys = list(mllm_query.keys())
    batch_idx = 0
    mllm_batch_inputs = []
    mllm_batch_keys = []
    for key in keys:
        mllm_batch_keys.append(key)
        image_obj = Image.open(mllm_query[key]["image_path"])
        mllm_batch_inputs.append({
            "prompt": prompt.format(mllm_query[key]["text_input"]),
            "multi_modal_data": {"image": image_obj}
        })
        if len(mllm_batch_inputs) >= mini_batch_size or key == keys[-1]:
            sta = batch_idx * mini_batch_size
            end = sta + len(mllm_batch_inputs)
            print('---MLLM Batch Inference from {} to {} , total {} ---'.format(sta, end, len(keys)))
            outputs = model.generate(mllm_batch_inputs, sampling_params)
            for k, o in zip(mllm_batch_keys, outputs):
                generated_text = o.outputs[0].text
                mllm_outputs[k] = generated_text
            batch_idx += 1
            mllm_batch_inputs = []
            mllm_batch_keys = []
    return mllm_outputs


@ray.remote
def clip_extact_feature_ray(clip_path, image_files, image_caption):
    model = CLIPModel.from_pretrained(clip_path)
    processor = AutoProcessor.from_pretrained(clip_path).image_processor
    tokenizer = AutoTokenizer.from_pretrained(clip_path)

    dataset = ImageFeatureDataset(image_files, image_caption, processor, tokenizer)
    image_dataloader = DataLoader(
        dataset, 
        batch_size=512, 
        collate_fn=dataset.collator, 
        shuffle=False, 
        num_workers=4
    )

    model = model.eval().to('cuda')
    image_embeddings = []
    caption_embeddings = []
    image_name_list = []

    with torch.no_grad():
        for batch in tqdm(image_dataloader, desc='Inference'):
            image_names = batch.pop('images_name')
            pixel_values = batch.pop('images_tensor').cuda()
            image_embedding = model.get_image_features(pixel_values)
            image_embedding = image_embedding.detach().cpu().numpy()    
            image_embeddings.append(image_embedding)

            for key in batch.keys():
                batch[key] = batch[key].cuda()
            caption_embedding = model.get_text_features(**batch)
            caption_embedding = caption_embedding.detach().cpu().numpy()
            caption_embeddings.append(caption_embedding)

            image_name_list.extend(image_names)
    
    image_embeddings = np.concatenate(image_embeddings, axis=0)
    caption_embeddings = np.concatenate(caption_embeddings, axis=0)
    return_dict = {
        'image_embedding': image_embeddings,
        'caption_embedding': caption_embeddings,
        'image_name_list': image_name_list
    }
    return return_dict


@ray.remote
def clip_rank_retrieval_ray(args, query_states, image_db_embedding, image_caption_embedidng, image_db_index):
    clip_model = CLIPModel.from_pretrained(args.clip_path).to('cuda')
    clip_model.eval()
    processor = AutoProcessor.from_pretrained(args.clip_path)
    image_db_index = image_db_index.tolist()
    image_db_embedding = torch.from_numpy(image_db_embedding).to("cuda").float()
    image_db_embedding = F.normalize(image_db_embedding, dim=1)
    image_caption_embedidng = torch.from_numpy(image_caption_embedidng).to("cuda").float()
    image_caption_embedidng = F.normalize(image_caption_embedidng, dim=1)

    clip_rank_results = defaultdict(dict)
    mini_batch_size = 128
    inference_data = []

    _sta_time = time.time()
    with torch.no_grad():
        query_keys = list(query_states.keys())
        for item_idx in tqdm(query_keys, total=len(query_keys), desc="[Stage1] CLIP Ranking"):
            tgt_img_desc_list = query_states[item_idx]['result']['stage1_reason']['json']['step2']
            # text_weight = float(query_states[item_idx]['result']['s1_plan']['json']['step3'][0])
            # image_weight = float(query_states[item_idx]['result']['s1_plan']['json']['step3'][1])

            ref_img = query_states[item_idx]['info']['ref_img']
            if ref_img is None:
                ref_img_obj = Image.new('RGB', (224, 224))
            else:
                ref_img_obj = Image.open(query_states[item_idx]['info']['ref_img_path'])

            if len(tgt_img_desc_list):
                for tgt_img_desc_idx, tgt_img_desc in enumerate(tgt_img_desc_list):
                    inference_data.append({
                        'item_idx': item_idx,
                        'num_of_tgt_img_desc': len(tgt_img_desc_list),
                        'tgt_img_desc': tgt_img_desc,
                        # 'text_weight': text_weight,
                        # 'image_weight': image_weight,
                        'ref_img': ref_img,
                        'ref_img_obj': ref_img_obj
                    })
            else:
                inference_data.append({
                    'item_idx': item_idx,
                    'num_of_tgt_img_desc': 1,
                    'tgt_img_desc': query_states[item_idx]['info']['inst'],
                    # 'text_weight': text_weight,
                    # 'image_weight': image_weight,
                    'ref_img': ref_img,
                    'ref_img_obj': ref_img_obj
                })

            if len(inference_data) >= mini_batch_size or item_idx == query_keys[-1]:
                batch_inference_data = inference_data
                inference_data = []
                batch_item_ids = [item['item_idx'] for item in batch_inference_data]
                batch_num_of_tgt_img_desc = [item['num_of_tgt_img_desc'] for item in batch_inference_data]
                batch_tgt_img_desc = [item['tgt_img_desc'] for item in batch_inference_data]
                # batch_text_weight = [item['text_weight'] for item in batch_inference_data]
                # batch_image_weight = [item['image_weight'] for item in batch_inference_data]
                batch_ref_imgs = [item['ref_img'] for item in batch_inference_data]
                batch_ref_img_objs = [item['ref_img_obj'] for item in batch_inference_data]
                
                text_inputs = processor.tokenizer(
                    batch_tgt_img_desc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    # max_length=77
                )
                
                text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}
                text_embeddings = clip_model.get_text_features(**text_inputs)
                text_embeddings = F.normalize(text_embeddings, dim=1).float()

                image_inputs = processor.image_processor(batch_ref_img_objs,  return_tensors='pt')['pixel_values'].to("cuda")
                image_embeddings = clip_model.get_image_features(pixel_values=image_inputs)
                image_embeddings = F.normalize(image_embeddings, dim=1).float()

                text_similarity = torch.matmul(text_embeddings, image_db_embedding.T)
                image_similarity = torch.matmul(text_embeddings, image_caption_embedidng.T)
                # similarity = text_similarity * torch.tensor(batch_image_weight)[:, None].cuda() + \
                #     image_similarity * torch.tensor(batch_text_weight)[:, None].cuda()
                similarity = text_similarity * (1-args.tau) + image_similarity * args.tau
                # similarity = text_similarity
                # similarity = text_similarity
                similarity = similarity.cpu()

                # for batch_idx, ref_img in enumerate(batch_ref_imgs):
                #     ref_image_index = image_db_index.index(ref_img)
                #     similarity[batch_idx, ref_image_index] = 0
                
                # sorted_indices = torch.argsort(similarity, descending=True)

                for batch_idx, (item_idx, num_of_tgt_img_desc) in enumerate(zip(batch_item_ids, batch_num_of_tgt_img_desc)):
                    if 'similarity' not in clip_rank_results[item_idx].keys():
                        clip_rank_results[item_idx]['similarity'] = similarity[batch_idx]
                    else:
                        clip_rank_results[item_idx]['similarity'] += similarity[batch_idx]
                    clip_rank_results[item_idx]['num_of_tgt_img_desc'] = num_of_tgt_img_desc
        
        for item_idx in tqdm(clip_rank_results.keys()):
            num_of_tgt_img_desc = clip_rank_results[item_idx].pop('num_of_tgt_img_desc')
            clip_rank_results[item_idx]['similarity'] /= num_of_tgt_img_desc
            clip_rank_results[item_idx]['sorted_indices'] = torch.argsort(clip_rank_results[item_idx]['similarity'], descending=True).tolist()
            clip_rank_results[item_idx]['similarity'] = clip_rank_results[item_idx]['similarity'][clip_rank_results[item_idx]['sorted_indices']].tolist()
            clip_rank_results[item_idx]['sorted_names'] = [image_db_index[i] for i in clip_rank_results[item_idx]['sorted_indices']]
            # clip_rank_results[item_idx]['similarity'] = clip_rank_results[item_idx]['similarity'].tolist()
    _end_time = time.time()
    _time_interval = _end_time - _sta_time
    return (clip_rank_results, {"time": _time_interval, "num_task": len(query_keys)})
    # pipe.send(clip_rank_results)
    # pipe.close()