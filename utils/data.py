import torch
import os
import json
from PIL import Image
from torch.utils.data import Dataset


class ImageFeatureDataset(Dataset):
    def __init__(self, image_files, image_caption, processor, tokenizer):
        self.image_files = image_files
        self.image_caption = image_caption
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name, img_path = self.image_files[idx]
        image_caption = self.image_caption[img_name]
        image = Image.open(img_path).convert("RGB")
        image = self.processor(image, return_tensors='pt')['pixel_values'][0]
        return image, img_name, image_caption

    def collator(self, batch):
        batch_image = [item[0] for item in batch]
        batch_image_name = [item[1] for item in batch]
        batch_image_caption = [item[2] for item in batch]

        batch_text_input = self.tokenizer(
            batch_image_caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            # max_length=77
        )

        batch_input = {}
        if all(x.shape == batch_image[0].shape for x in batch_image):
            batch_image = torch.stack(batch_image, dim=0)
        batch_input['images_tensor'] = batch_image
        batch_input['images_name'] = batch_image_name
        batch_input.update(batch_text_input)

        return batch_input
    

def load_CIRR(dataset_path):
    # with open(os.path.join(dataset_path, 'image_splits/split.rc2.train.json'), 'r') as f:
    #     train_img_dict = json.load(f)
    with open(os.path.join(dataset_path, 'image_splits/split.rc2.val.json'), 'r') as f:
        val_img_dict = json.load(f)
    with open(os.path.join(dataset_path, 'image_splits/split.rc2.test1.json'), 'r') as f:
        test_img_dict = json.load(f)

    image_files = []
    image_dict = {}
    # for img_name, img_path in train_img_dict.items():
    #     img_path = os.path.join(dataset_path, img_path.replace('.///', ''))
    #     train_img_dict[img_name] = img_path
    #     image_files.append((img_name, img_path))
    #     image_dict[img_name] = img_path
    for img_name, img_path in val_img_dict.items():
        img_path = os.path.join(dataset_path, img_path.replace('.///', ''))
        val_img_dict[img_name] = img_path
        image_dict[img_name] = img_path
        image_files.append((img_name, img_path))
    for img_name, img_path in test_img_dict.items():
        img_path = os.path.join(dataset_path, img_path.replace('.///', ''))
        test_img_dict[img_name] = img_path
        image_files.append((img_name, img_path))
        image_dict[img_name] = img_path

    # train_data = json.load(open(os.path.join(dataset_path, 'captions/cap.rc2.train.json'), 'r'))
    val_data = json.load(open(os.path.join(dataset_path, 'captions/cap.rc2.val.json'), 'r'))
    test_data = json.load(open(os.path.join(dataset_path, 'captions/cap.rc2.test1.json'), 'r'))

    def extract_ref_instruct_tgt(item):
        ref_img = item['reference']
        if 'target_hard' in item.keys():
            tgt_img = item['target_hard']
        else:
            tgt_img = None
        instruction = item['caption'].strip()
        img_set = item['img_set']['members']

        # We find that a relative instruction is empty. 
        # To prevent exceptions and errors, we replace it with "the same image".
        if len(instruction) == 0:
            instruction = 'the same image'
        
        return {
            'pairid': item['pairid'],
            'ref': ref_img,
            'tgt': tgt_img,
            'instruction': instruction,
            'img_set': img_set
        }
    
    # train_data = [extract_ref_instruct_tgt(item) for item in train_data]
    val_data = [extract_ref_instruct_tgt(item) for item in val_data][:2400] # only for testing
    test_data = [extract_ref_instruct_tgt(item) for item in test_data]
    image_dict_split = {
        # 'train': train_img_dict,
        'val': val_img_dict,
        'test': test_img_dict,
        'all': image_dict
    }
    data = {
        # 'train': train_data,
        'val': val_data,
        'test': test_data,
    }
    return data, image_dict_split


def load_FashionIQ(dataset_path, category):
    assert category in ['dress', 'shirt', 'toptee']

    with open(os.path.join(dataset_path, 'image_splits/split.{}.val.json').format(category), 'r') as f:
        val_img_list = json.load(f)
        val_img_dict = dict()
        for image_name in val_img_list:
            val_img_dict[image_name] = os.path.join(dataset_path, 'images/{}.png'.format(image_name))

    with open(os.path.join(dataset_path, 'captions/cap.{}.val.json').format(category), 'r') as f:
        val_data = json.load(f)

    def extract_ref_instruct_tgt(item):
        ref_img = item['candidate']
        tgt_img = item['target']
        instruction = item['captions'][0].strip('.? ') + ' and ' + item['captions'][1].strip('.? ') 
        return {
            'ref': ref_img,
            'tgt': tgt_img,
            'instruction': instruction,
        }
    val_data = [extract_ref_instruct_tgt(item) for item in val_data]

    image_dict_split = {
        'val': val_img_dict,
        'all': val_img_dict  # for caption preprocessing
    }
    data = {
        'val': val_data,
    }
    return data, image_dict_split


def load_CIRCO(dataset_path):
    images_files = os.listdir(os.path.join(dataset_path, 'unlabeled2017'))
    images_files = sorted(images_files)

    # all images are used for CIRCO
    val_img_dict = {}
    for img_name in images_files:
        img_path = os.path.join(dataset_path, 'unlabeled2017', img_name)
        val_img_dict[img_name] = img_path
    
    with open(os.path.join(dataset_path, 'captions/val.json'), 'r') as f:
        val_data = json.load(f)
    
    with open(os.path.join(dataset_path, 'captions/test.json'), 'r') as f:
        test_data = json.load(f)
    
    def extract_ref_instruct_tgt(item):
        idx = item['id']
        ref_img = item['reference_img_id'] 
        ref_img = str(ref_img).zfill(12) + '.jpg'
        if 'target_img_id' not in item.keys():
            tgt_img = None
            gt_img_ids = None
        else:
            tgt_img = item['target_img_id']
            tgt_img = str(tgt_img).zfill(12) + '.jpg'
            gt_img_ids = item['gt_img_ids']
            gt_img_ids = [str(gt_img_id).zfill(12) for gt_img_id in gt_img_ids]
        instruction = item['relative_caption']
        return {
            'id': idx,
            'ref': ref_img,
            'tgt': tgt_img,
            'instruction': instruction,
            'gt_img_ids': gt_img_ids,
        }

    val_data = [extract_ref_instruct_tgt(item) for item in val_data]
    test_data = [extract_ref_instruct_tgt(item) for item in test_data]

    image_dict_split = {
        'val': val_img_dict,
        'test': val_img_dict,
        'all': val_img_dict  # for caption preprocessing
    }
    data = {
        'val': val_data,
        'test': test_data
    }
    return data, image_dict_split


def load_MSCOCO(dataset_path):
    import pandas as pd
    dataframe = pd.read_csv(os.path.join(dataset_path, 'test_5k_mscoco_2014.csv'))

    images_files = os.listdir(os.path.join(dataset_path, 'images'))
    images_files = sorted(images_files)

    # select test split
    dataframe = dataframe[dataframe['split']=='test']
    test_img_dict = dict()
    for image_name in images_files:
        test_img_dict[image_name] = os.path.join(dataset_path, 'images', image_name)

    def extract_ref_instruct_tgt(row):
        ref_img = None
        tgt_img = row['filename']
        caption_list = eval(row['raw'])
        sample_list = []
        for caption in caption_list:
            sample_list.append({
                'ref': ref_img,
                'tgt': tgt_img,
                'instruction': caption,
            })
        return sample_list
    
    test_data = []
    for _, row in dataframe.iterrows():
        test_data += extract_ref_instruct_tgt(row)
    
    image_dict_split = {
        'test': test_img_dict,
        'all': test_img_dict  # for caption preprocessing
    }
    data = {
        'test': test_data
    }
    return data, image_dict_split


def load_Flickr30K(dataset_path):
    import pandas as pd
    dataframe = pd.read_csv(os.path.join(dataset_path, 'test_1k_flickr.csv'))

    images_files = os.listdir(os.path.join(dataset_path, 'images'))
    images_files = sorted(images_files)

    # select test data split from df
    dataframe = dataframe[dataframe['split']=='test']
    test_img_dict = dict()
    for image_name in images_files:
        test_img_dict[image_name] = os.path.join(dataset_path, 'images', image_name)

    def extract_ref_instruct_tgt(row):
        ref_img = None
        tgt_img = row['filename']
        caption_list = eval(row['raw'])
        sample_list = []
        for caption in caption_list:
            sample_list.append({
                'ref': ref_img,
                'tgt': tgt_img,
                'instruction': caption,
            })
        return sample_list

    test_data = []
    for _, row in dataframe.iterrows():
        test_data += extract_ref_instruct_tgt(row)

    image_dict_split = {
        'test': test_img_dict,
        'all': test_img_dict  # for caption preprocessing
    }
    data = {
        'test': test_data
    }
    return data, image_dict_split


def load_VisDial(dataset_path):
    with open(os.path.join(dataset_path, 'Search_Space_val_50k.json'), 'r') as f:
        val_imgs = json.load(f)

    val_img_dict = dict()
    for img_name in val_imgs:
        val_img_dict[img_name.split('/')[-1]] = os.path.join(dataset_path, img_name)
    
    with open(os.path.join(dataset_path, 'VisDial_v1.0_queries_val.json'), 'r') as f:
        original_val_data = json.load(f)
    
    val_data = []
    for item in original_val_data:
        ref_img = None
        tgt_img = item['img'].split('/')[-1]
        dialog_list = item['dialog']
        val_data.append({
            'ref': ref_img,
            'tgt': tgt_img,
            'dialog_list': dialog_list,
        })

    image_dict_split = {
        'val': val_img_dict,
        'all': val_img_dict  # for caption preprocessing
    }
    data = {
        'val': val_data
    }
    return data, image_dict_split
    

def load_data(dataset_name, dataset_path):
    assert dataset_name in [
        'CIRR', 'CIRR-subset', 
        'FashionIQ-dress', 'FashionIQ-shirt', 'FashionIQ-toptee', 
        'CIRCO', 
        'MSCOCO', 'Flickr30K',
        'VisDial',
    ]
    data_split = dataset_name.split('-')
    if len(data_split) == 2:
        _dataset_name, _split = data_split
    elif len(data_split) == 1:
        _dataset_name = data_split[0]
        _split = None
    dataset_path = os.path.join(dataset_path, _dataset_name)
    if _dataset_name == 'CIRR':
        return load_CIRR(dataset_path)
    elif _dataset_name == 'FashionIQ':
        return load_FashionIQ(dataset_path, _split)
    elif _dataset_name == 'CIRCO':
        return load_CIRCO(dataset_path)
    elif _dataset_name == 'MSCOCO':
        return load_MSCOCO(dataset_path)
    elif _dataset_name == 'Flickr30K':
        return load_Flickr30K(dataset_path)
    elif _dataset_name == 'VisDial':
        return load_VisDial(dataset_path)
    else:
        raise NotImplementedError
