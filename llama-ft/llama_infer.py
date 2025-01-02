import os
import torch
import argparse
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json
from tqdm import tqdm
import random
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
torch.multiprocessing.set_start_method('spawn', force=True)

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "chinese": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n给定音高、音量、年龄、性别、语气等信息以及文本，运用情感分析的技巧，用中文自然语言描述。"
        "注意必须生动且多样化地描述，不需要描述肢体动作或心理状态，不要重复input内容。\n\n"
        "###Input：\n{labels}\n\n### Response:"
    ),
    "english": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nGiven information such as pitch, volume, age, gender, tone, category and text, describe using English natural language with the techniques of emotional analysis."
        "Make sure the description is vivid and diverse, without the need to describe physical actions or psychological states, and avoid repeating the input content.\n\n"
        "###Input：\n{labels}\n\n### Response:"

    )
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.language = args.language
        self.ckpt_path = args.ckpt_path
        self.testdata = json.load(open(args.json_path))
        self.keyindex = list(self.testdata.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=False, trust_remote_code=True)

    def __len__(self):
        return len(self.testdata)

    def __getitem__(self, index):
        user_tokens=[195]
        assistant_tokens=[196]
        key = self.keyindex[index]
        value = self.testdata[key]
        tags = value['labels'].strip().split('\t')

        # shuffle tags
        random.shuffle(tags)
        value['labels'] = '\t'.join(tags)

        prompt = '<reserved_106>' + PROMPT_DICT[self.language].format_map(value) + '<reserved_107>'
        inputs = self.tokenizer(prompt, return_tensors='pt')

        return inputs, key, prompt, value['labels']


def extract(input_text, language):
    if language=='chinese':
        if "“" not in input_text:
            return None
        else:
            start_index = input_text.find("“")
            end_index = input_text.find("”", start_index+1)
            return input_text[start_index:end_index]
    else:
        if input_text.count("\"") < 2:
            return None
        else:
            start_index = input_text.find("\"")
            end_index = input_text.rfind("\"")
            return input_text[start_index+1:end_index]

def inference_on_device(args, tokenizer, device, dataloader):

    language = args.language
    output_path = args.output_path
    error_path = args.error_path
    ckpt_path = args.ckpt_path

    model = AutoPeftModelForCausalLM.from_pretrained(
        ckpt_path, 
        revision="v2.0",
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        )

    # model = model.quantize(4)
    model.to(device)
    top_p = 0.3
    temperature = 0.7
    count = 0
    test_result  = {}
    error = {}
    output_path = output_path[:-4]+str(device)+'.json'
    with torch.no_grad():  
        for inputs, keys, prompts, labels in tqdm(dataloader):
            # unwrap the input
            inputs = {key: value[0].to(device, dtype=torch.long) for key, value in inputs.items()}
            # inputs = {key: value[0].to(device, dtype=torch.bfloat16) for key, value in inputs.items()}
            key = keys[0]
            prompt = prompts[0]
            info = {}
            labels = labels[0]

            try:
                pred = model.sample(**inputs, max_new_tokens=128, repetition_penalty=1.1, do_sample=True, use_cache=True, temperature=temperature, top_p=top_p)
                generation = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[len(prompt):]
                ct = 0

                while (extract(generation, language) is None or (len(generation)-len(extract(generation, language))<3)) and ct<5:
                    pred = model.sample(**inputs, max_new_tokens=128, repetition_penalty=1.1, do_sample=True, use_cache=True, temperature=temperature, top_p=top_p)
                    generation = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[len(prompt):]
                    ct+=1

                if ct==5:
                    with open(error_path, 'a') as file:
                        file.write(key+'\n')
                    continue

                info['labels'] = labels
                info['llama-instruction'] = generation

                test_result[key] = info
                count+=1

            except Exception as e:
                with open(error_path, 'a') as file:
                    file.write(key+'\n')
                # raise e
            
            if count % 1000 ==0 or count==50:
                json.dump(test_result, open(output_path, 'w'), indent=4, ensure_ascii=False)
        json.dump(test_result, open(output_path, 'w'), indent=4, ensure_ascii=False)
        

def main(args):
    devices = list(map(int, args.devices.split(',')))

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, revision="v2.0", use_fast=False, trust_remote_code=True)
    inferset = Dataset(args)
    num_devices = len(devices)
    subset_size = len(inferset) // num_devices
    subsets = [ Subset(inferset, range(i * subset_size, (i + 1) * subset_size)) for i in range(num_devices) ]
    data_loaders = [ DataLoader(subset, batch_size=1) for subset in subsets ]
    
    processes = []
    for i in range(num_devices):
        device_num = devices[i]
        device = torch.device(f'cuda:{device_num}')
        # device = torch.device('cpu')
        p = torch.multiprocessing.Process(target=inference_on_device, args=(args, tokenizer, device, data_loaders[i]))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default = 'english')
    parser.add_argument('--devices', type=str, default = '0')
    parser.add_argument('--json_path', type=str, default = './example_libritts.json')
    parser.add_argument('--output_path', type=str, default = './inference_libritts.json')
    parser.add_argument('--error_path', type=str, default = './error.txt')
    parser.add_argument('--ckpt_path', type=str, default = './finetuned-llama')
    
    args = parser.parse_args()
    
    main(args)
