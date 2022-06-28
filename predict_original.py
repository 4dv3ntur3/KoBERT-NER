from lib2to3.pgen2 import token
import os
import logging
import argparse
from memory_profiler import profile
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from utils import init_logger, load_tokenizer, get_labels

# os.environ["LRU_CACHE_CAPACITY"] = "1"

from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    training_params = torch.load(os.path.join(pred_config.model_dir, 'training_params.bin'))

    args = training_params['training_args']
    label_lst = training_params['label_lst']

    return args, label_lst


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(pred_config.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            # line = line.strip()
            # words = line.split()
            # lines.append(words)
            lines.append(line)
    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []


    all_input_tokens = []
    # for words in lines:
    #     tokens = []
    #     slot_label_mask = []
    #     for word in words:
    #         word_tokens = tokenizer.tokenize(word)
    #         if not word_tokens:
    #             word_tokens = [unk_token]  # For handling the bad-encoded word
    #         tokens.extend(word_tokens)
            
    #         # use the real label id for all tokens of the word
    #         slot_label_mask.extend([0] * (len(word_tokens)))

    #     all_input_tokens.append(tokens)

    #     # Account for [CLS] and [SEP]
    #     special_tokens_count = 2
    #     if len(tokens) > args.max_seq_len - special_tokens_count:
    #         tokens = tokens[: (args.max_seq_len - special_tokens_count)]
    #         slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

    #     # Add [SEP] token
    #     tokens += [sep_token]
    #     token_type_ids = [sequence_a_segment_id] * len(tokens)
    #     slot_label_mask += [pad_token_label_id]

    #     # Add [CLS] token
    #     tokens = [cls_token] + tokens
    #     token_type_ids = [cls_token_segment_id] + token_type_ids
    #     slot_label_mask = [pad_token_label_id] + slot_label_mask


    #     input_ids = tokenizer.convert_tokens_to_ids(tokens)


    #     # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    #     attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    #     # Zero-pad up to the sequence length.
    #     print("====================", args.max_seq_len)
    #     padding_length = args.max_seq_len - len(input_ids)

    #     input_ids = input_ids + ([pad_token_id] * padding_length)

    #     attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    #     token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    #     slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

    
    kiwi = Kiwi()
    # print(lines[0])
    split_sentences = kiwi.split_into_sents(lines[0])
    inputs = [sentence.text for sentence in split_sentences]

    for input in inputs:

        tokenized = tokenizer(input)

        input_ids = input_ids + ([pad_token_id] * padding_length)

        padding_length = args.max_seq_len - len(input_ids)

        
        all_input_ids.extend(tokenized['input_ids'][:args.max_seq_len])
        all_attention_mask.extend(tokenized['attention_mask'][:args.max_seq_len])
        all_token_type_ids.extend(tokenized['token_type_ids'][:args.max_seq_len])
        slot_label_mask = [-100]
        slot_label_mask.extend([0] * (len(tokenized['input_ids'][:args.max_seq_len])-2))
        slot_label_mask.extend([-100])
        all_slot_label_mask.append(slot_label_mask)


    # slot_label_mask = [-100]
    # slot_label_mask.extend([0] * (len(tokenized['input_ids'][0][:args.max_seq_len])-2))
    # slot_label_mask.extend([-100])

    # slot_label_mask = slot_label_mask + ([pad_token_label_id])

    # print(len(input_ids))

    # all_input_ids.append(tokenized['input_ids'][0][:args.max_seq_len])
    # all_attention_mask.append(tokenized['attention_mask'][0][:args.max_seq_len])
    # all_token_type_ids.append(tokenized['token_type_ids'][0][:args.max_seq_len])
    # all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    # print(all_input_ids)
    # print(all_attention_mask)
    # print(all_slot_label_mask)
    # quit()
    
    # all_input_tokens = tokenizer.convert_ids_to_tokens(all_input_ids, skip_special_tokens=True)

    for batches in all_input_tokens:
        decoded = tokenizer.convert_ids_to_tokens(batches)
        all_input_tokens.append(decoded)

    # print(all_input_tokens)

    # print(tokenized)
    # print(len(tokenized['input_ids'][0]))

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return all_input_ids, all_attention_mask, all_slot_label_mask, all_input_tokens
    # return dataset

def predict(pred_config):
    # load model and args
    args, label_lst = get_args(pred_config)
    device = get_device(pred_config)

    model = load_model(pred_config, args, device)
    logger.info(args)

    args.max_seq_len = 512

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)

    # tokenized = tokenizer(lines)
    # print(tokenized)
    # print(len(tokenized['input_ids'][0]))

    lines = read_input_file(pred_config)

    all_input_ids, all_attention_mask, all_slot_label_mask, all_input_tokens = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # Predict
    # sampler = SequentialSampler(dataset)
    # data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask_ = None
    preds = None

    with torch.no_grad():
        inputs = {"input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": None}

        outputs = model(**inputs)
        logits = outputs[0]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            all_slot_label_mask_ = all_slot_label_mask.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            all_slot_label_mask_ = np.append(all_slot_label_mask, all_slot_label_mask.detach().cpu().numpy(), axis=0)
# 
    

    # for batch in tqdm(data_loader, desc="Predicting"):
    #     batch = tuple(t.to(device) for t in batch)
    #     with torch.no_grad():
    #         inputs = {"input_ids": batch[0],
    #                 "attention_mask": batch[1],
    #                 "labels": None}
    #         if args.model_type != "distilkobert":
    #             inputs["token_type_ids"] = batch[2]
    #         outputs = model(**inputs)
    #         logits = outputs[0]

    #         if preds is None:
    #             preds = logits.detach().cpu().numpy()
    #             all_slot_label_mask = batch[3].detach().cpu().numpy()
    #         else:
    #             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    #             all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask_[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])

    # Save output file in ./model/{model_dir}
    model_dir = pred_config.model_dir.split(os.sep)[-1]
    input_fn = pred_config.input_file.split(os.sep)[-1]
    output_dir = os.path.join('./pred_out', model_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_fn = pred_config.output_file

    if output_fn is None:
        output_fn = input_fn
        output_path = os.path.join(output_dir, output_fn)
    else:
        output_path = output_fn

    # print([all_input_tokens])
    print(preds_list)


    with open(output_path, "w", encoding="utf-8") as f:
        for words, preds in zip([all_input_tokens], preds_list):
            line = ""
            for word, pred in zip(words, preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)

            f.write("{}\n".format(line.strip()))

    logger.info("Prediction Done!")

# os.environ["LRU_CACHE_CAPACITY"] = "1"




if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='./long__test.txt', type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default=None, type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)
