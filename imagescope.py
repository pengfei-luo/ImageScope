import os
import torch
import json
import pickle
from utils.function import create_logger
from datetime import datetime
from copy import deepcopy
from json_repair import repair_json
import multiprocessing as mp
import re
import argparse
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils.inference_entrypoint_ray import (
    llm_load_and_inference_ray,
    mllm_load_and_inference_ray,
    mllm_inference_for_caption_ray,
    clip_extact_feature_ray,
    clip_rank_retrieval_ray,
    stage3_large_mllm_inference_qa_ray,
    stage3_large_mllm_inference_batch_cmp_ray,
)
from utils.function import get_num_of_gpu
from utils.data import load_data
import ray
import time


class ImageScope(object):
    def __init__(self, args=None):
        super(ImageScope, self).__init__()
        # Load dataset and image split
        self.args = args
        self.data, self.image_dict_split = load_data(args.dataset, args.dataset_path)
        
        clip_version_str = self.args.clip_path.split('/')[-1].split('-')[:4]
        self.clip_version = '-'.join(clip_version_str)
        print("Current CLIP Version: ", self.clip_version)

        self.current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir_name = f"runs/{args.dataset}_{args.split}/{self.clip_version}-{args.run_name}_{self.current_timestamp}"
        os.makedirs(self.dir_name, exist_ok=False)
        self.logger = create_logger(os.path.join(self.dir_name, 'output.log'))
        
        with mp.Pool(1) as pool: 
            self.num_of_gpu = pool.apply(get_num_of_gpu)

        if (self.args.dataset == 'CIRR' and self.args.subset):
            if self.args.dataset == 'CIRR':
                self.top_k = min(self.args.top_k, len(self.data['test'][0]['img_set']) - 1)
            self.logger.debug(f'Reset top k to {self.top_k}, because subset is set')
        else:
            self.top_k = self.args.top_k
        self.load_prompt()
        self.states = defaultdict(dict)

        _args_dict = vars(self.args)
        print(f"{json.dumps(_args_dict, indent=4)}")
        self.logger.info('Arguments')
        self.logger.info(f"{json.dumps(_args_dict, indent=4)}")
        self.logger.info('------------------------------------\n')
        self.logger.info(f'Current Time: {self.current_timestamp}. Start to run.')

    def load_prompt(self):
        # Load prompt
        ds_name = self.args.dataset
        with open(f'./prompts/{ds_name}/prompt1_stage1_reasoner.txt', 'r') as f:
            self.stage1_reasoner_prompt = f.read()
        with open(f'./prompts/{ds_name}/prompt2_stage2_reasoner.txt', 'r') as f:
            self.stage2_reasoner_prompt = f.read()
        with open(f'./prompts/{ds_name}/caption.txt', 'r') as f:
            self.caption_prompt = f.read()
        with open(f'./prompts/{ds_name}/prompt3_stage3_evaluator.txt', 'r') as f:
            self.stage3_evaluator_prompt = f.read()
        
        # Save prompt for this run
        with open(f"{self.dir_name}/prompt1_stage1_reasoner.txt", "w") as f:
            f.write(self.stage1_reasoner_prompt)
        with open(f"{self.dir_name}/prompt2_stage2_reasoner.txt", "w") as f:
            f.write(self.stage2_reasoner_prompt)
        with open(f"{self.dir_name}/caption.txt", "w") as f:
            f.write(self.caption_prompt)
        with open(f"{self.dir_name}/prompt3_stage3_evaluator.txt", "w") as f:
            f.write(self.stage3_evaluator_prompt)


    def load_image_db(self, filter_dict):
        # Load processed image features and caption features
        image_embedding = np.load(f'./image_db/{self.args.dataset}/{self.clip_version}/image_embedding.npy')
        image_caption_embedding = np.load(f'./image_db/{self.args.dataset}/{self.clip_version}/image_caption_emebdding.npy')
        image_name = np.load(f'./image_db/{self.args.dataset}/{self.clip_version}/image_name_list.npy')

        # Filter image database
        if filter_dict is not None:
            image_name = image_name.tolist()
            filtered_image_name = []
            filtered_image_embedding = []
            filtered_image_caption_embedding = []
            for img_name, _ in tqdm(filter_dict.items(), desc='Loading Image DB'):
                idx = image_name.index(img_name)
                embedding = image_embedding[idx]
                filtered_image_name.append(img_name)
                filtered_image_embedding.append(embedding)
                filtered_image_caption_embedding.append(image_caption_embedding[idx])
        
            image_name = np.array(filtered_image_name)
            image_embedding = np.array(filtered_image_embedding)
            image_caption_embedding = np.array(filtered_image_caption_embedding)
        
        print(f'Loaded Image DB Size: {len(image_name)}')
        return image_embedding, image_caption_embedding, image_name
        
    def check_and_perpare_image_database(self):
        # Check the necessary files
        os.makedirs(f'./image_db/{self.args.dataset}/{self.clip_version}', exist_ok=True)

        # Generate the image caption if not exists
        if not os.path.exists(f'./image_db/{self.args.dataset}/image_caption.json'):
            print('Generate Image Caption...')
            mllm_query = defaultdict(dict)
            for image_name, image_path in self.image_dict_split['all'].items():
                mllm_query[image_name]['text_input'] = self.caption_prompt
                mllm_query[image_name]['image_path'] = image_path

            # Request MLLM inference
            keys = list(mllm_query.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {k: mllm_query[k] for k in key_chunk}
                    task = mllm_inference_for_caption_ray.options(num_gpus=1, num_cpus=8).remote(self.args.img_cap_model_path, query_chunk)
                    tasks.append(task)
                    time.sleep(4)
                results = ray.get(tasks)
            mllm_outputs = dict()
            for result in results:
                mllm_outputs.update(result) 
           
            image_caption_dict = {}
            for image_name, image_desc in mllm_outputs.items():
                image_caption_dict[image_name] = image_desc

            # Save image caption
            with open(f'./image_db/{self.args.dataset}/image_caption.json', 'w') as f:
                json.dump(image_caption_dict, f, indent=4)
        else:
            # If the image caption exists, load it
            with open(f'./image_db/{self.args.dataset}/image_caption.json', 'r') as f:
                image_caption_dict = json.load(f)
            
        self.image_caption_dict = image_caption_dict

        # Check if the image embedding and image caption embedding exist
        if not os.path.exists(f'./image_db/{self.args.dataset}/{self.clip_version}/image_embedding.npy') or \
            not os.path.exists(f'./image_db/{self.args.dataset}/{self.clip_version}/image_name_list.npy') or \
            not os.path.exists(f'./image_db/{self.args.dataset}/{self.clip_version}/image_caption_emebdding.npy'):
            print('Extact Image and Caption Embedding...')

            with open(f'./image_db/{self.args.dataset}/image_caption.json', 'r') as f:
                image_caption_dict = json.load(f)
            
            # Request CLIP to extact image embedding and caption embedding
            image_files = {idx: (image_name, image_path) for idx, (image_name, image_path) in enumerate(self.image_dict_split['all'].items())}
            keys = list(image_files.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {j: image_files[k] for j, k in enumerate(key_chunk)}
                    task = clip_extact_feature_ray.options(num_gpus=1, num_cpus=8).remote(self.args.clip_path, query_chunk, image_caption_dict)
                    tasks.append(task)
                results = ray.get(tasks)
            
            image_embedding = []
            image_caption_emebdding = []
            image_name_list = []
            for result in results:
                image_embedding.append(result['image_embedding'])
                image_caption_emebdding.append(result['caption_embedding'])
                image_name_list.append(result['image_name_list'])
            image_embedding = np.concatenate(image_embedding, axis=0)
            image_caption_emebdding = np.concatenate(image_caption_emebdding, axis=0)
            image_name_list = np.concatenate(image_name_list, axis=0)

            # Save image embedding
            np.save(f'./image_db/{self.args.dataset}/{self.clip_version}/image_embedding.npy', image_embedding)
            np.save(f'./image_db/{self.args.dataset}/{self.clip_version}/image_caption_emebdding.npy', image_caption_emebdding)
            np.save(f'./image_db/{self.args.dataset}/{self.clip_version}/image_name_list.npy', image_name_list)
            
        print('All files are satisfied.')

    def save_states(self, format=['pkl'], save_response_text=False, save_clip_topk=50, round_idx=None):
        print('Saving States...')
        # current_states = deepcopy(self.states)
        current_states = pickle.loads(pickle.dumps(self.states))
        for item_idx in current_states.keys():
            if 'result' in current_states[item_idx]:
                if 'stage1_retrieve' in current_states[item_idx]['result']:
                    current_states[item_idx]['result']['stage1_retrieve']['similarity'] = current_states[item_idx]['result']['stage1_retrieve']['similarity'][:save_clip_topk]
                    current_states[item_idx]['result']['stage1_retrieve']['sorted_names'] = current_states[item_idx]['result']['stage1_retrieve']['sorted_names'][:save_clip_topk]
                    current_states[item_idx]['result']['stage1_retrieve']['sorted_indices'] = current_states[item_idx]['result']['stage1_retrieve']['sorted_indices'][:save_clip_topk]
                if 's2_rerank' in current_states[item_idx]['result']:
                    current_states[item_idx]['result']['s2_rerank']['sorted_names'] = current_states[item_idx]['result']['s2_rerank']['sorted_names'][:save_clip_topk]
        
        if 'pkl' in format:
            if round_idx is None:
                pkl_filename = os.path.join(self.dir_name, 'states.pkl')
            else:
                pkl_filename = os.path.join(self.dir_name, f'states_round_{round_idx}.pkl')
            with open(pkl_filename, 'wb') as f:
                pickle.dump(current_states, f)
        if 'json' in format:
            json_filename = os.path.join(self.dir_name, 'states.json')
            with open(json_filename, 'w') as f:
                json.dump(current_states, f, indent=2)
        print('Saving States... Done')

    def calculate_recall(self, rank_result, k_list=[1,2,3,5,10,20,50,100]):
        recalls_at_K = {}
        for k in k_list:
            recalls_at_K[k] = np.mean([1 if _rnk <= k else 0 for _rnk in rank_result]).item()
        return recalls_at_K

    def stage1_reason(self, multi_round=False):
        
        # 1. Prepare LLM query
        llm_query = []
        for item_idx, item in enumerate(self.task_data):
            item_plan_prompt = deepcopy(self.stage1_reasoner_prompt)
            item_plan_prompt = item_plan_prompt.replace('[[INSTRUCTION]]', item['instruction'])
            if multi_round:
                item_plan_prompt = item_plan_prompt.replace('[[REF_IMAGE_DESC]]', self.states[item_idx]['info']['last_round_tgt_desc'].strip())
            else:
                item_plan_prompt = item_plan_prompt.replace('[[REF_IMAGE_DESC]]', self.states[item_idx]['info']['ref_img_desc'].strip())
            llm_query.append(item_plan_prompt)
        
        # 2. Request LLM
        chunk_size = math.ceil(len(llm_query) / self.num_of_gpu)
        query_chunks = [llm_query[i:i + chunk_size] for i in range(0, len(llm_query), chunk_size)]
        with ray.init():
            tasks = []
            for i, chunk in enumerate(query_chunks):
                task = llm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.llm_path, chunk)
                tasks.append(task)
                time.sleep(6)
            results = ray.get(tasks)
        llm_outputs = [item for (item_list, _) in results for item in item_list]
        
        # Log task time information
        task_time = [d for (_, d) in results]
        for _idx, d in enumerate(task_time):
            _iter_avg_time = d['time'] / d['num_task']
            self.logger.debug(f"LLM Inference {_idx}: Wall Time {d['time']:.4f}, Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
        total_wall_time = sum([d['time'] for d in task_time])
        total_num_task = sum([d['num_task'] for d in task_time])
        iter_avg_time = total_wall_time / total_num_task
        self.logger.debug(f"LLM Inference Overall: Wall Time {total_wall_time:.4f}, Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        # 3. Collect LLM reasoning results
        for item_idx, output in enumerate(llm_outputs):
            generated_text = output.outputs[0].text
            num_input_tokens = len(output.prompt_token_ids)
            num_output_tokens = len(output.outputs[0].token_ids)

            if 'result' not in self.states[item_idx].keys():
                self.states[item_idx]['result'] = dict()

            self.states[item_idx]['result']['stage1_reason'] = dict()
            self.states[item_idx]['result']['stage1_reason']['num_input_tokens'] = num_input_tokens
            self.states[item_idx]['result']['stage1_reason']['num_output_tokens'] = num_output_tokens

            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, generated_text, re.DOTALL)

            if match is None:
                json_pattern_braces = r"(\{[\s\S]*?\})"
                match = re.search(json_pattern_braces, generated_text, re.DOTALL)

            if match:
                json_str = match.group(1)
                json_obj = json.loads(repair_json(json_str))
                if not isinstance(json_obj, dict):
                    self.logger.debug(f"{item_idx}: {json_str}")
                self.states[item_idx]['result']['stage1_reason']['json'] = json_obj
                if "step2" not in self.states[item_idx]['result']['stage1_reason']['json'].keys():
                    self.logger.debug(f"{item_idx}: No target descriptions found. Using defatul description.")
                    self.logger.debug(generated_text)
                    self.logger.debug('_-------_')
                    self.logger.debug(json_obj)
                    self.states[item_idx]['result']['stage1_reason']['json']['step2'] = [self.states[item_idx]['info']['ref_img_desc']]
                    continue
                if len(self.states[item_idx]['result']['stage1_reason']['json']['step2']) == 0:
                    self.states[item_idx]['result']['stage1_reason']['json']['step2'] = [self.states[item_idx]['info']['ref_img_desc']]
                    self.logger.debug(f"{item_idx}: No target descriptions found. Using defatul description.")
            else:
                # Dummy input for the error
                self.states[item_idx]['result']['stage1_reason']['json'] = dict()
                self.states[item_idx]['result']['stage1_reason']['json']['step1'] = [["Modification", self.states[item_idx]['info']['inst']]]  
                self.states[item_idx]['result']['stage1_reason']['json']['step2'] = [self.states[item_idx]['info']['ref_img_desc']]
                self.logger.debug(f"{llm_query[item_idx]}")
                self.logger.debug(f'ERROR: {generated_text}')

        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. Target Descriptions Generated.")


    def stage1_retrieve(self):
        if 'CIRCO' in self.args.dataset:
            image_embedding, image_caption_embedding, image_name = self.load_image_db(None)
        else:
            image_embedding, image_caption_embedding, image_name = self.load_image_db(self.image_split)

        # 1. Clip ranking
        keys = list(self.states.keys())
        chunk_size = math.ceil(len(keys) / self.num_of_gpu)
        key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
        with ray.init():
            tasks = []
            for i, key_chunk in enumerate(key_chunks):
                query_chunk = {k: self.states[k] for k in key_chunk}
                # clip_path, states, query_keys, image_db_embedding, image_caption_embedidng, image_db_index
                task = clip_rank_retrieval_ray.options(num_gpus=1, num_cpus=8).remote(self.args, query_chunk, image_embedding, image_caption_embedding, image_name)
                tasks.append(task)
                time.sleep(5)
            results = ray.get(tasks)
        clip_rank_results = {}
        for (result, _) in results:
            clip_rank_results.update(result) 

        # Log time information
        task_time = [d for (_, d) in results]
        for _idx, d in enumerate(task_time):
            _iter_avg_time = d['time'] / d['num_task']
            self.logger.debug(f"CLIP Inference {_idx}: Wall Time {d['time']:.4f}, Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
        total_wall_time = sum([d['time'] for d in task_time])
        total_num_task = sum([d['num_task'] for d in task_time])
        iter_avg_time = total_wall_time / total_num_task
        self.logger.debug(f"CLIP Inference Overall: Wall Time {total_wall_time:.4f}, Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        # 3. Save the results to states
        for item_idx in clip_rank_results.keys():
            self.states[item_idx]['result']['stage1_retrieve'] = dict()
            for key in clip_rank_results[item_idx].keys():
                self.states[item_idx]['result']['stage1_retrieve'][key] = clip_rank_results[item_idx][key][:15000]  # we only save top 15000 candidates to save space

        if (self.args.dataset == 'CIRR' and self.args.subset):
            for item_idx in self.states.keys():
                subset_img_name = self.states[item_idx]['meta']['img_set']
                index_in_results = []
                for _img_name in subset_img_name:
                     if _img_name == self.states[item_idx]['info']['ref_img']:
                         continue
                     _idx = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'].index(_img_name)
                     index_in_results.append(_idx)
                subset_sorted_names = [self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][_idx] for _idx in index_in_results]
                subset_similarity = [self.states[item_idx]['result']['stage1_retrieve']['similarity'][_idx] for _idx in index_in_results]
                subset_sorted_indices = [self.states[item_idx]['result']['stage1_retrieve']['sorted_indices'][_idx] for _idx in index_in_results]
                
                similarity_array = np.array(subset_similarity)
                sorted_indices = np.argsort(similarity_array)[::-1]
                subset_similarity = similarity_array[sorted_indices].tolist()
                subset_sorted_names = [subset_sorted_names[i] for i in sorted_indices]
                subset_sorted_indices = [subset_sorted_indices[i] for i in sorted_indices]

                self.states[item_idx]['result']['stage1_retrieve']['sorted_names'] = subset_sorted_names
                self.states[item_idx]['result']['stage1_retrieve']['similarity'] = subset_similarity
                self.states[item_idx]['result']['stage1_retrieve']['sorted_indices'] = subset_sorted_indices
        
        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. Clip ranking done.")

        if self.args.split == 'valid' or (self.args.split == 'test' and self.args.dataset not in ['CIRR','CIRCO']):
            for item_idx in self.states.keys():
                tgt_img = self.states[item_idx]['info']['tgt_img']
                if tgt_img in self.states[item_idx]['result']['stage1_retrieve']['sorted_names']:
                    tgt_img_rank = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'].index(tgt_img) + 1
                else:
                    tgt_img_rank = len(self.states[item_idx]['result']['stage1_retrieve']['sorted_names']) + 1
                self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank'] = tgt_img_rank
                
            tgt_img_ranks = [self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank'] for item_idx in self.states.keys()]
            metrics_recall = self.calculate_recall(tgt_img_ranks)
            for k in metrics_recall.keys():
                self.logger.info(f"Recall@{k}: {metrics_recall[k] * 100}")

    def stage2_verify(self):
        # 1. Prepare the llm query for reasoner
        llm_query = []
        for item_idx in self.states.keys():
            atomic_inst = self.states[item_idx]['result']['stage1_reason']['json']['step1']
            atomic_inst_text = '\n'

             # Here we filter out the invalid atomic instrucion, i.e., the length of the element is not 2.
            inst_idx = 1
            for element in atomic_inst:
                if len(element) != 2:
                    continue
                inst_type, inst_content = element
                atomic_inst_text += f"   ({inst_idx}) {inst_type}: {inst_content}\n"
                inst_idx += 1
            atomic_prompt = deepcopy(self.stage2_reasoner_prompt)
            atomic_prompt = atomic_prompt.replace('[[INSTRUCTION]]', self.states[item_idx]['info']['inst'])
            atomic_prompt = atomic_prompt.replace('[[ATOMIC_INST]]', atomic_inst_text)
            llm_query.append(atomic_prompt)

        # 2. Request llm
        chunk_size = math.ceil(len(llm_query) / self.num_of_gpu)
        query_chunks = [llm_query[i:i + chunk_size] for i in range(0, len(llm_query), chunk_size)]
        with ray.init():
            tasks = []
            for i, chunk in enumerate(query_chunks):
                task = llm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.llm_path, chunk)
                tasks.append(task)
                time.sleep(5)
            results = ray.get(tasks)
        llm_outputs = [item for (item_list, _) in results for item in item_list]
        
        # Log task time information
        task_time = [d for (_, d) in results]
        for _idx, d in enumerate(task_time):
            _iter_avg_time = d['time'] / d['num_task']
            self.logger.debug(f"LLM Inference {_idx}: Wall Time {d['time']:.4f}, Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
        total_wall_time = sum([d['time'] for d in task_time])
        total_num_task = sum([d['num_task'] for d in task_time])
        iter_avg_time = total_wall_time / total_num_task
        self.logger.debug(f"LLM Inference Overall: Wall Time {total_wall_time:.4f}, Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        # Collect the results
        for item_idx, output in enumerate(llm_outputs):
            generated_text = output.outputs[0].text
            self.states[item_idx]['result']['s2_atomic'] = dict()
            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, generated_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                json_obj = json.loads(repair_json(json_str))
                
                # Remove the invalid atomic proposition
                try:
                    valid_atomic_proposition = []
                    for q_idx, q_element in enumerate(json_obj['step2']):
                        if isinstance(q_element, list) and len(q_element) == 2:
                            valid_atomic_proposition.append(q_element)
                    json_obj['step2'] = valid_atomic_proposition
                except:
                    self.logger.debug(f'STEP2 {item_idx} | ERROR: {generated_text}')
                    
                self.states[item_idx]['result']['s2_atomic']['json'] = json_obj
            else:
                self.logger.debug(f'{item_idx} | ERROR: {generated_text}')

        
        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. Atomic propositions and questions generated.")


    def stage2_rerank(self):
        # 1. Prepare MLLM query
        mllm_query = defaultdict(dict)
        for item_idx in self.states.keys():
            if 'json' not in self.states[item_idx]['result']['s2_atomic'].keys() or \
                'step2' not in self.states[item_idx]['result']['s2_atomic']['json'].keys():
                print(item_idx, ' error when generating json objs, please refer to log.')
                continue

            mllm_query[item_idx]['text_inputs'] = list()
            mllm_query[item_idx]['ref_img_path'] = self.states[item_idx]['info']['ref_img_path'] 
            for q_idx, q_element in enumerate(self.states[item_idx]['result']['s2_atomic']['json']['step2']):
                if not isinstance(q_element, list):
                    continue
                elif len(q_element) != 2:
                    continue
                q_text, q_ans = q_element
                mllm_query[item_idx]['text_inputs'].append(
                    f"Answer the question with Yes or No. {q_text}"
                )

            mllm_query[item_idx]['top_k_ranked_candidates'] = dict()

            # If restored states and have verify results
            already_verify_candidates = []
            if 's2_verify' in self.states[item_idx]['result'].keys() and \
                'candidate_satisfy_value' in self.states[item_idx]['result']['s2_verify'].keys():
                already_verify_candidates = list(range(len(self.states[item_idx]['result']['s2_verify']['candidate_satisfy_value'])))
                
            for candidate_idx in range(self.top_k):
                if candidate_idx in already_verify_candidates:
                    continue
                candidate_image_name = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][candidate_idx]
                candidate_image_path = self.image_split[candidate_image_name]
                mllm_query[item_idx]['top_k_ranked_candidates'][candidate_idx] = {
                    'image_name': candidate_image_name,
                    'image_path': candidate_image_path
                }

        # 2. Request MLLM for verification
        if len(mllm_query[0]['top_k_ranked_candidates']):
            keys = list(mllm_query.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {k: mllm_query[k] for k in key_chunk}
                    task = mllm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.mllm_path, query_chunk)
                    tasks.append(task)
                    time.sleep(6)
                results = ray.get(tasks)
            mllm_outputs = dict()
            for result, _ in results:
                mllm_outputs.update(result) 

            # Log task time information
            task_time = [d for (_, d) in results]
            for _idx, d in enumerate(task_time):
                _iter_avg_time = d['time'] / d['num_task']
                self.logger.debug(f"MLLM Inference {_idx}: Wall Time {d['time']:.4f}, Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
            total_wall_time = sum([d['time'] for d in task_time])
            total_num_task = sum([d['num_task'] for d in task_time])
            iter_avg_time = total_wall_time / total_num_task
            self.logger.debug(f"MLLM Inference Overall: Wall Time {total_wall_time:.4f}, Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        # 3. Collect results
        for item_idx in tqdm(mllm_query.keys(), desc='Verifying atomic proposition', total=len(mllm_query)):
            if 's2_verify' not in self.states[item_idx]['result'].keys():
                self.states[item_idx]['result']['s2_verify'] = dict()
            
            candidates_correct_value = []
            candidates_response = []
            candidates_response_ori = []
            for candidate_idx in range(self.top_k):
                try:
                    already_have_candidate_num = 0
                    if 's2_verify' in self.states[item_idx]['result'].keys() and \
                        'candidate_response_ori' in self.states[item_idx]['result']['s2_verify'].keys():
                        already_have_candidate_num = len(self.states[item_idx]['result']['s2_verify']['candidate_response_ori'])
                    if already_have_candidate_num == 0:
                        candidate_response_ori = mllm_outputs[item_idx][candidate_idx]
                    else:
                        if candidate_idx < already_have_candidate_num:
                            candidate_response_ori = self.states[item_idx]['result']['s2_verify']['candidate_response_ori'][candidate_idx]
                        else:
                            candidate_response_ori = mllm_outputs[item_idx][candidate_idx - already_have_candidate_num]       

                    candidate_response = ['True' if 'yes' in response.strip().lower() else 'False' for response in candidate_response_ori]
                    candidate_atomic_ground_truth =  [_q_ans for _q_text, _q_ans in self.states[item_idx]['result']['s2_atomic']['json']['step2']]
                    if len(candidate_atomic_ground_truth) != len(candidate_response):
                        self.logger.debug('Number of responses does not match')
                        self.logger.debug(f'{item_idx}')
                        self.logger.debug(f'Num of generate: {len(candidate_response)}, Num of ground truth: {len(candidate_atomic_ground_truth)}')
                        self.logger.debug(mllm_outputs[item_idx][candidate_idx])
                    candidate_atomic_verify_result = sum([
                        candidate_response[i] == candidate_atomic_ground_truth[i] for i in range(min(len(candidate_atomic_ground_truth), len(candidate_response)))
                    ])
                    candidates_correct_value.append(candidate_atomic_verify_result)
                    candidates_response.append(candidate_response)
                    candidates_response_ori.append(candidate_response_ori)
                except:
                    self.logger.debug(f'Error {item_idx}')
                    self.logger.debug(self.states[item_idx]['result']['s2_atomic']['json'])
                    candidates_correct_value.append(0)
                    candidates_response.append(['False'] * len(candidate_response))
                                  
            self.states[item_idx]['result']['s2_verify']['candidate_satisfy_value'] = candidates_correct_value
            self.states[item_idx]['result']['s2_verify']['candidate_response'] = candidates_response
            self.states[item_idx]['result']['s2_verify']['candidate_response_ori'] = candidates_response_ori

        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. Reranking finished.")

        # 4. Rerank the candidate images based on value
        for item_idx in tqdm(self.states.keys(), desc='Reranking', total=len(self.states)):
            if 's2_rerank' not in self.states[item_idx]['result'].keys():
                self.states[item_idx]['result']['s2_rerank'] = dict()
            top_k_candidates = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][:self.top_k]
            if 's2_verify' not in self.states[item_idx]['result'].keys():
                self.logger.error(f'{item_idx} | s2_verify found for {item_idx}, skipping...')
                self.states[item_idx]['result']['s2_rerank']['sorted_names'] = deepcopy(self.states[item_idx]['result']['stage1_retrieve']['sorted_names'])
                continue

            top_k_candidates_atomic_value = self.states[item_idx]['result']['s2_verify']['candidate_satisfy_value']
            sorted_candidates_with_values = sorted(
                zip(top_k_candidates, (-np.array(top_k_candidates_atomic_value)).tolist()),
                key=lambda x: x[1],
            )
            top_k_candidates_sorted, top_k_candidates_atomic_value_sorted = zip(*sorted_candidates_with_values)
            self.states[item_idx]['result']['s2_rerank']['sorted_names'] = deepcopy(self.states[item_idx]['result']['stage1_retrieve']['sorted_names'])
            self.states[item_idx]['result']['s2_rerank']['sorted_names'][:self.top_k] = top_k_candidates_sorted

        if self.args.split == 'valid' or (self.args.split == 'test' and self.args.dataset not in ['CIRR','CIRCO']):
            for item_idx in self.states.keys():
                tgt_img = self.states[item_idx]['info']['tgt_img']
                if tgt_img in self.states[item_idx]['result']['s2_rerank']['sorted_names']:
                    tgt_img_rank = self.states[item_idx]['result']['s2_rerank']['sorted_names'].index(tgt_img) + 1
                else:
                    tgt_img_rank = len(self.states[item_idx]['result']['s2_rerank']['sorted_names']) + 1
                self.states[item_idx]['result']['s2_rerank']['tgt_img_rank'] = tgt_img_rank

            rank_results = [self.states[item_idx]['result']['s2_rerank']['tgt_img_rank'] for item_idx in self.states.keys()]
            metrics_recall = self.calculate_recall(rank_results)
            for k in metrics_recall.keys():
                self.logger.info(f"Recall@{k}: {metrics_recall[k] * 100}")

        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. Atomic proposition verification finished")
    
    def stage3_evaluate(self):
        # 1. Prepare MLLM evaluation query
        if 'stage3_eval' not in self.states[0]['result'].keys():
            mllm_query = defaultdict(dict)
            for item_idx in self.states.keys():
                mllm_query[item_idx]['top_candidate'] = list()
                mllm_query[item_idx]['inst'] = self.states[item_idx]['info']['inst']
                mllm_query[item_idx]['ref_img_path'] = self.states[item_idx]['info']['ref_img_path']
                
                text_input = deepcopy(self.stage3_evaluator_prompt)
                text_input = text_input.replace('[[INSTRUCTION]]', mllm_query[item_idx]['inst'].strip(' .').lower())

                for idx in range(self.args.alpha):
                    candidate_img_name = self.states[item_idx]['result']['s2_rerank']['sorted_names'][idx]
                    candidate_img_path = self.image_split[candidate_img_name]

                    mllm_query[item_idx]['top_candidate'].append({
                        'image_name': candidate_img_name,
                        'image_path': candidate_img_path
                    })

                mllm_query[item_idx]['text_input'] = text_input

        
            # 2. Split query, request MLLM and collect result
            keys = list(mllm_query.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {k: mllm_query[k] for k in key_chunk}
                    task = stage3_large_mllm_inference_batch_cmp_ray.options(
                        num_gpus=1, 
                        num_cpus=8
                    ).remote(self.args.eval_mllm_path, query_chunk)
                    tasks.append(task)
                    time.sleep(5)
                results = ray.get(tasks)
            mllm_outputs = dict()
            for result, _ in results:
                mllm_outputs.update(result)
            
            # Log task time information
            task_time = [d for (_, d) in results]
            for _idx, d in enumerate(task_time):
                _iter_avg_time = d['time'] / d['num_task']
                self.logger.debug(f"MLLM Inference {_idx}: Wall Time {d['time']:.4f}, Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
            
            num_turn = len(task_time[0]['turn_time'])
            for turn_idx in range(num_turn):
                for _idx, d in enumerate(task_time):
                    _turn_avg_time = d['turn_time'][turn_idx] / d['turn_task'][turn_idx]
                    self.logger.debug(f"MLLM Inference {_idx} Turn {turn_idx+1}: Wall Time {d['turn_time'][turn_idx]:.4f}, Number Task {d['turn_task'][turn_idx]}, Avg {_turn_avg_time:.4f}")
                
            total_wall_time = sum([d['time'] for d in task_time])
            total_num_task = sum([d['num_task'] for d in task_time])
            iter_avg_time = total_wall_time / total_num_task
            self.logger.debug(f"MLLM Inference Overall: Wall Time {total_wall_time:.4f}, Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        # 3. Evaluate the top ranked candidates
        for item_idx in tqdm(self.states.keys(), desc='[Stege3] Evaluating'):
            if 'stage3_eval' not in self.states[item_idx]['result'].keys():
                self.states[item_idx]['result']['stage3_eval'] = dict()
                self.states[item_idx]['result']['stage3_eval']['outputs'] = mllm_outputs[item_idx]
            sorted_names = deepcopy(
                self.states[item_idx]['result']['s2_rerank']['sorted_names']
            )
            
            evaluated_correct_idx = []
            evaluated_sorted_names = []
            for turn_idx, turn_result_dict in enumerate(self.states[item_idx]['result']['stage3_eval']['outputs'][:self.args.alpha]):
                decision = turn_result_dict['decision']
                if decision:
                    evaluated_correct_idx.append(turn_idx)
            if len(evaluated_correct_idx):
                for correct_idx in evaluated_correct_idx:
                    img = sorted_names.pop(correct_idx)
                    evaluated_sorted_names.append(img)
                evaluated_sorted_names = evaluated_sorted_names + sorted_names
            else:
                evaluated_sorted_names = sorted_names
            self.states[item_idx]['result']['stage3_eval']['sorted_names'] = evaluated_sorted_names 


        if self.args.split == 'valid' or (self.args.split == 'test' and self.args.dataset not in ['CIRR','CIRCO']):
            for item_idx in self.states.keys():
                tgt_img = self.states[item_idx]['info']['tgt_img']
                if tgt_img in self.states[item_idx]['result']['stage3_eval']['sorted_names']:
                    tgt_img_rank = self.states[item_idx]['result']['stage3_eval']['sorted_names'].index(tgt_img) + 1
                else:
                    tgt_img_rank = len(self.states[item_idx]['result']['stage3_eval']['sorted_names']) + 1
                self.states[item_idx]['result']['stage3_eval']['tgt_img_rank'] = tgt_img_rank
            
            rank_results = [self.states[item_idx]['result']['stage3_eval']['tgt_img_rank'] for item_idx in self.states.keys()]
            metrics_recall = self.calculate_recall(rank_results)
            for k in metrics_recall.keys():
                self.logger.info(f"Recall@{k}: {metrics_recall[k] * 100}")
        
        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. [Stage3] Evaluation is finished.")


    def start(self):
        if self.args.split == 'valid':
            self.task_data = self.data['val']
            self.image_split = self.image_dict_split['val']
        else:
            self.task_data = self.data['test']
            self.image_split = self.image_dict_split['test']
        
        if self.args.restore_states is not None:
            if os.path.exists(self.args.restore_states):
                self.logger.info(f"Restoring status from {self.args.restore_states}")
                self.states = pickle.load(open(self.args.restore_states, 'rb'))
            else:
                raise FileNotFoundError(f"Status file {self.args.restore_status} does not exist")
        
        self.check_and_perpare_image_database()

        for item_idx, item in enumerate(self.task_data):
            if 'CIRR' in self.args.dataset:
                self.states[item_idx]['meta'] = dict()
                self.states[item_idx]['meta']['pairid'] = item['pairid'] 
                self.states[item_idx]['meta']['img_set'] = item['img_set']
            elif 'CIRCO' in self.args.dataset:
                self.states[item_idx]['meta'] = dict()
                self.states[item_idx]['meta']['pairid'] = item['id'] 

            # save to states
            self.states[item_idx]['info'] = dict()
            self.states[item_idx]['info']['ref_img'] = item['ref']
            self.states[item_idx]['info']['ref_img_path'] = self.image_split[item['ref']] if item['ref'] is not None else None
            self.states[item_idx]['info']['ref_img_desc'] = self.image_caption_dict[item['ref']] if item['ref'] is not None else "A blank image."
            self.states[item_idx]['info']['inst'] = item['instruction']

            if self.args.split == 'test' and self.args.dataset in ['CIRR', 'CIRCO']:
                pass
            else:
                self.states[item_idx]['info']['tgt_img'] = item['tgt']
                self.states[item_idx]['info']['tgt_img_path'] = self.image_split[item['tgt']]
       
        ###############################
        ###                         ###
        ###############################
        if 's1' in self.args.stages:
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 0):
                self.stage1_reason()
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 1):
                self.stage1_retrieve()
        if 's2' in self.args.stages:
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 2):
                self.stage2_verify()
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 3):
                self.stage2_rerank()
        if 's3' in self.args.stages:
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 4):
                self.stage3_evaluate()

        ###############################
        ###                         ###
        ###############################

        if self.args.split == 'test' and self.args.dataset in ['CIRR', 'CIRCO']:
            self.logger.info(f"Genereating test results...")
            test_results = dict()
            defatut_submit_top_k = 50
            result_file_name = f"{self.current_timestamp}_{self.args.dataset}_{self.args.split}_stage1_retrieve.json"
            if 'CIRR' in self.args.dataset:
                if self.args.subset:
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall_subset'
                    defatut_submit_top_k = 3
                    result_file_name = f"{self.current_timestamp}_{self.args.dataset}_subset_{self.args.split}_clip.json"
                else:
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall'
            

            for item_idx in self.states.keys():
                pairid = self.states[item_idx]['meta']['pairid']
                top_50_prediction = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][:defatut_submit_top_k]
                test_results[pairid] = top_50_prediction
                if self.args.dataset == 'CIRCO':
                    for idx, pred in enumerate(test_results[pairid]):
                            # remove .jpg and zeros in th beginging
                            pred = pred[:-4].lstrip('0')
                            test_results[pairid][idx] = pred
            
            self.logger.info(f"Saving test results to {os.path.join(self.dir_name, result_file_name)}")
            with open(os.path.join(self.dir_name, result_file_name), 'w') as f:
                json.dump(test_results, f)

            if "s2_rerank" in self.states[item_idx]['result'].keys():
                test_results = dict()
                if self.args.dataset == 'CIRR':
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall' if not self.args.subset else 'recall_subset'
                for item_idx in self.states.keys():
                    pairid = self.states[item_idx]['meta']['pairid']
                    top_50_prediction = self.states[item_idx]['result']['s2_rerank']['sorted_names'][:defatut_submit_top_k]
                    test_results[pairid] = top_50_prediction
                    if self.args.dataset == 'CIRCO':
                        for idx, pred in enumerate(test_results[pairid]):
                            # remove .jpg and zeros in th beginging
                            pred = pred[:-4].lstrip('0')
                            test_results[pairid][idx] = pred
                rerank_file_name = f"{self.current_timestamp}_{self.args.dataset}_{self.args.split}_stage2_rerank.json"
                self.logger.info(f"Saving rerank results to {os.path.join(self.dir_name, rerank_file_name)}")
                with open(os.path.join(self.dir_name, rerank_file_name), 'w') as f:
                    json.dump(test_results, f)

            if "stage3_eval" in self.states[item_idx]['result'].keys():
                test_results = dict()
                if self.args.dataset == 'CIRR':
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall' if not self.args.subset else 'recall_subset'
                for item_idx in self.states.keys():
                    pairid = self.states[item_idx]['meta']['pairid']
                    top_50_prediction = self.states[item_idx]['result']['stage3_eval']['sorted_names'][:defatut_submit_top_k]
                    test_results[pairid] = top_50_prediction
                    if self.args.dataset == 'CIRCO':
                        for idx, pred in enumerate(test_results[pairid]):
                            # remove .jpg and zeros in th beginging
                            pred = pred[:-4].lstrip('0')
                            test_results[pairid][idx] = pred
                rerank_file_name = f"{self.current_timestamp}_{self.args.dataset}_{self.args.split}_stage3_eval.json"
                self.logger.info(f"Saving rerank results to {os.path.join(self.dir_name, rerank_file_name)}")
                with open(os.path.join(self.dir_name, rerank_file_name), 'w') as f:
                    json.dump(test_results, f)
            
        elif self.args.split == 'valid':
            max_logged_case = 10
            cnt = 0
            for item_idx in self.states.keys():
                # previous_rank = self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank']
                # verified_rank = self.states[item_idx]['result']['s2_rerank']['tgt_img_rank']
                # rank_change = previous_rank - verified_rank
                if True:
                # if previous_rank <= self.topk and rank_change <= 0 and verified_rank != 1:
                    cnt += 1
                    self.logger.debug(f'-------{item_idx}-------')
                    self.logger.debug("# Query")
                    self.logger.debug("## Instruction\n - {}".format(self.states[item_idx]['info']['inst']))
                    self.logger.debug("## Reference Image Description\n - {}".format(self.states[item_idx]['info']['ref_img_desc']))
                    self.logger.debug("Ref Img Path: {}".format(self.states[item_idx]['info']['ref_img_path']))
                    if self.args.split == 'valid':
                        self.logger.debug("Tgt Img Path: {}".format(self.states[item_idx]['info']['tgt_img_path']))
                        self.logger.debug("Tgt Img Desc: {}".format(self.image_caption_dict[self.states[item_idx]['info']['tgt_img']]))
                    self.logger.debug('## Target Image Description')
                    if 'json' in self.states[item_idx]['result']['stage1_reason'].keys():
                        # self.logger.debug(json.dumps(self.states[item_idx]['result']['stage1_reason']['json'], indent=2))
                        for d_idx, d in enumerate(self.states[item_idx]['result']['stage1_reason']['json']['step1']):
                            self.logger.debug(f" ({d_idx+1}) {d}")
                    self.logger.debug('## Verification questions')
                    if 's2_atomic' in self.states[item_idx]['result'].keys() and 'json' in self.states[item_idx]['result']['s2_atomic'].keys():
                        # self.logger.debug(json.dumps(self.states[item_idx]['result']['s2_atomic']['json'], indent=2))
                        for q_idx, (q, a) in enumerate(self.states[item_idx]['result']['s2_atomic']['json']['step2']):
                            self.logger.debug(f" ({q_idx+1}) {q} Ground Truth: {a}.")
                    self.logger.debug('# Candidates')
                    for candidate_idx in range(self.top_k):
                        img_name = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][candidate_idx]
                        img_path = self.image_split[img_name]
                        img_desc = self.image_caption_dict[img_name]
                        similarity_score = self.states[item_idx]['result']['stage1_retrieve']['similarity'][candidate_idx]
                        is_ground_truth = candidate_idx + 1 == self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank']
                        ground_truth_flag = '  <- Ground truth' if is_ground_truth else ''
                        self.logger.debug(f"## Candidate {candidate_idx+1} {ground_truth_flag}")
                        self.logger.debug(" - Similarity: {:.4f}".format(similarity_score))
                        self.logger.debug(" - Image description: {}".format(img_desc))
                        if 's2_verify' in self.states[item_idx]['result'].keys():
                            self.logger.debug(" - Response of Questions: {}".format(self.states[item_idx]['result']['s2_verify']['candidate_response'][candidate_idx]))
                            _gt = [x for _, x in self.states[item_idx]['result']['s2_atomic']['json']['step2']]
                            self.logger.debug("  Ground truth: {}".format(_gt))
                            self.logger.debug('  Correct: {}'.format(self.states[item_idx]['result']['s2_verify']['candidate_satisfy_value'][candidate_idx]))
                            self.logger.debug(" - Candidate Image {}".format(img_path))
                    self.logger.debug("Previous reank {}".format(self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank']))
                    # self.logger.debug("New rerank {}".format(self.states[item_idx]['result']['s2_rerank']['tgt_img_rank']))
                    # self.logger.debug("Rank change: {}".format(rank_change))
                    self.logger.debug(f'-------{item_idx}-------')
                if cnt >= max_logged_case:
                    break
    
    def start_dialog(self):
        if self.args.split == 'valid':
            self.task_data = self.data['val']
            self.image_split = self.image_dict_split['val']
        else:
            self.task_data = self.data['test']
            self.image_split = self.image_dict_split['test']
        
        if self.args.restore_states is not None:
            if os.path.exists(self.args.restore_states):
                self.logger.info(f"Restoring status from {self.args.restore_states}")
                self.states = pickle.load(open(self.args.restore_states, 'rb'))
            else:
                raise FileNotFoundError(f"Status file {self.args.restore_status} does not exist")
        
        self.check_and_perpare_image_database()

        ###############################
        ###                         ###
        ###############################
        task_data = deepcopy(self.task_data)
        total_round = len(task_data[0]['dialog_list'])
        current_round_task = dict()
        k_list = [1,2,3,4,5,10,20,25,50]
        stage_hits_rate = {_: dict() for _ in [1,2,3]}
        for stage_idx in stage_hits_rate.keys():
            stage_hits_rate[stage_idx] = {
                k: [0] * len(task_data) for k in k_list
            }
        

        for round_idx in range(1, total_round):
            self.logger.info(f"Starting round {round_idx}")
            
            states = {}
            current_round_task_data = []
            for item_idx in range(len(task_data)):
                if round_idx == 1:
                    last_round_tgt_desc = task_data[item_idx]['dialog_list'][0]
                else:
                    if 'json' in self.states[item_idx]['result']['stage1_reason'].keys() and len(self.states[item_idx]['result']['stage1_reason']['json']['step2']):
                        last_round_tgt_desc = self.states[item_idx]['result']['stage1_reason']['json']['step2'][0]
                
                current_round_task_data.append({
                    'ref': None,
                    'tgt': task_data[item_idx]['tgt'],
                    'instruction': task_data[item_idx]['dialog_list'][round_idx],
                    'last_round_tgt_desc': last_round_tgt_desc,
                })

                if item_idx <= 5:
                    s = f"{item_idx}: " +  "last_round_tgt_desc: " + last_round_tgt_desc +  ". instruction: " + task_data[item_idx]['dialog_list'][round_idx]
                    self.logger.info(s)

            self.states = defaultdict(dict)
            self.task_data = current_round_task_data

            for item_idx, item in enumerate(self.task_data):
                self.states[item_idx]['info'] = dict()
                self.states[item_idx]['info']['ref_img'] = item['ref']
                self.states[item_idx]['info']['ref_img_path'] = self.image_split[item['ref']] if item['ref'] is not None else None
                self.states[item_idx]['info']['ref_img_desc'] = self.image_caption_dict[item['ref']] if item['ref'] is not None else "A blank image."
                self.states[item_idx]['info']['inst'] = item['instruction']
                self.states[item_idx]['info']['last_round_tgt_desc'] = item['last_round_tgt_desc']
                self.states[item_idx]['info']['tgt_img'] = item['tgt']
                self.states[item_idx]['info']['tgt_img_path'] = self.image_split[item['tgt']]
        
            ###############################
            ###                         ###
            ###############################
            if 's1' in self.args.stages:
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 0):
                    self.stage1_reason(multi_round=True)
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 1):
                    self.stage1_retrieve()
                
                self.logger.info(f"Stage 1 Hits rate.")
                tgt_img_ranks = [self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank'] for item_idx in self.states.keys()]
                for k in k_list:
                    for rnk_idx, rnk in enumerate(tgt_img_ranks):
                        if stage_hits_rate[1][k][rnk_idx]:
                            continue
                        else:
                            if rnk <= k:
                                stage_hits_rate[1][k][rnk_idx] += 1
                    self.logger.info(f"Hits@{k}: {sum(stage_hits_rate[1][k]) / len(stage_hits_rate[1][k])}")


            if 's2' in self.args.stages:
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 2):
                    self.stage2_verify()
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 3):
                    self.stage2_rerank()

                self.logger.info(f"Stage 2 Hits rate.")
                tgt_img_ranks = [self.states[item_idx]['result']['s2_rerank']['tgt_img_rank'] for item_idx in self.states.keys()]
                for k in k_list:
                    for rnk_idx, rnk in enumerate(tgt_img_ranks):
                        if stage_hits_rate[2][k][rnk_idx]:
                            continue
                        else:
                            if rnk <= k:
                                stage_hits_rate[2][k][rnk_idx] += 1
                    self.logger.info(f"Hits@{k}: {sum(stage_hits_rate[2][k]) / len(stage_hits_rate[2][k])}")

            if 's3' in self.args.stages:
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 4):
                    self.stage3_evaluate()

                self.logger.info(f"Stage 3 Hits rate.")
                tgt_img_ranks = [self.states[item_idx]['result']['stage3_eval']['tgt_img_rank'] for item_idx in self.states.keys()]
                for k in k_list:
                    for rnk_idx, rnk in enumerate(tgt_img_ranks):
                        if stage_hits_rate[3][k][rnk_idx]:
                            continue
                        else:
                            if rnk <= k:
                                stage_hits_rate[3][k][rnk_idx] += 1
                    self.logger.info(f"Hits@{k}: {sum(stage_hits_rate[3][k]) / len(stage_hits_rate[3][k])}")
            
            self.save_states(round_idx=round_idx)
            
            ###############################
            ###                         ###
            ###############################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--llm_path', type=str, default=None, help='Reasoner model path')
    parser.add_argument('--mllm_path', type=str, default=None, help='Verifier model path')
    parser.add_argument('--img_cap_model_path', type=str, default=None, help='Captioner model path')
    parser.add_argument('--eval_mllm_path', type=str, default=None, help='Evaluator model path')
    parser.add_argument('--clip_path', type=str, default=None, help='VLM path')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset folder path')
    
    parser.add_argument('--run_name', type=str, default='default')
    
    parser.add_argument('--tau', type=float, default=0.15)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--alpha', type=int, default=3)
    parser.add_argument('--stages', nargs='+', choices=['s1', 's2', 's3'], default=['s1', 's2', 's3'])

    parser.add_argument('--dataset', type=str, choices=[
        'CIRR', 'CIRCO', 
        'FashionIQ-dress', 'FashionIQ-shirt', 'FashionIQ-toptee', 
        'MSCOCO', 'Flickr30K',
        'VisDial'])
    parser.add_argument('--subset', action='store_true', default=False, help='Use subset of the dataset, for CIRR only')
    
    # Use valid for FashionIQ and VisDial, test for CIRCO, CIRR, MSCOCO and Flickr30K
    parser.add_argument('--split', type=str, choices=["valid", "test"], default="valid")
    
    # The following arguments are for restoring the states from a previous run. Ignore if not needed.
    parser.add_argument('--restore_states', type=str, default=None)
    parser.add_argument('--stage_skip_num', type=int, choices=[0,1,2,3,4], default=0)
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()

    imagescope = ImageScope(args)
    if args.dataset == 'VisDial':
        imagescope.start_dialog()
    else:
        imagescope.start()

    