import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from utils import init_logger, load_tokenizer, get_labels

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
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
            line = line.strip()
            words = line.split()
            lines.append(words)

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
    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))
            
            # use the real label id for all tokens of the word
            slot_label_mask.extend([0] * (len(word_tokens)))

        # print(tokens)
        all_input_tokens.append(tokens) # 뭐지? 여기서는 뒤에 sep 안 붙는데 이 이후부터 계속 붙네?
        # append를 여기서 하는데 왜지??? 
        # print("=====================")
        # print(all_input_tokens)
        # 어차피 input_tokens 뒤에 SEP가 붙어서 +1이더라도 preds_list는 그거보다 1개 작으니까 (SEP 없음) 상관은 없는데
        # 뭐임??
        
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # print("=====================")
        # print(all_input_tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        # print("=====================")
        # print(all_input_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)

        input_ids = input_ids + ([pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    # print("=====================")
    # print(all_input_tokens)
    # quit()

    return dataset, all_input_tokens


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)

    model = load_model(pred_config, args, device)
    label_lst = get_labels(args)
    logger.info(args)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    dataset, all_input_tokens = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # print(all_input_tokens)
    # quit()

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    preds = None
    sms = None
    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None} # label이 None이므로 
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)

            print(len(outputs)) # 1

            logits = outputs[0] # loss? 
            # print(logits.size()) # 3, 50, 28 (B, S, num_classes)
            # print(logits.detach().cpu().numpy())

            # logits = linear output
            sm = torch.nn.functional.softmax(logits, dim=-1)
            sm = sm.detach().cpu().numpy()
            # print(sm)
            # print(sm.size()) # 3, 50, 28

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
                sms = sm
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
                sms = np.append(sms, axis=0)

    preds = np.argmax(preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    sms_list = [[] for _ in range(sms.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])
                sms_list[i].append(sms[i][j])

    ######### 정답 후보 태그 3개까지 출력 
    for k in range(len(sms_list)):

      temp = sms_list[k] #19 (Number of tokens)

      # print(temp)
      # print(len(temp))
      # print("=======================")

      for tidx in range(len(temp)):
        
        real_tok = all_input_tokens[k][tidx]
        tok = temp[tidx]
        ranked = np.argsort(tok)
        largest_indices = ranked[::-1][:3]

        print("rank for", real_tok, " started")

        for i in range(len(largest_indices)):
          idx = largest_indices[i]
          # print(idx)
          print(slot_label_map[idx], ": ", tok[idx])

        print("rank for", real_tok, " ended\n\n")

    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, preds in zip(all_input_tokens, preds_list): # 한 문장 
            
            line = ""
            
            word_end = False
            pii_word = ""
            
            for i, (word, pred) in enumerate(zip(words, preds)): # token별
                
                if i < len(preds)-1:
                    
                    if word_end == True:
                        pii_word = ""
                    
                    
                    after_word = words[i+1]
                    after_pred = preds[i+1]
                    
                    word = word.strip('#').strip()
                    
                    pii_word += word
                    
                    # print(pii_word)
                    
                    # 다음 단어에 #이 포함되어 있다면 이어지는 단어 
                    if '#' in after_word:

                        # 뒤 토큰과 한 단어이지만 무의미하므로 따로 쓴다
                        if after_pred == 'O':
                            
                            if pred == 'O':
                                # 둘 다 Other인데 한 단어이다
                                word_end = False
                                
                            else:
                                # 그러면 그냥 혼자 적기
                                line = line + "[{}:{}] ".format(pii_word, pred)
                            
                                word_end = True
                            

                            
                        # 뒤 토큰과 한 단어이고, 유의미해서 같이 적어야 함
                        else: 
                            word_end = False
                            
                    else:
                        word_end = True
                        
                        if pred == 'O':
                            line = line + pii_word + " "
                        
                        else:
                            line = line + "[{}:{}] ".format(pii_word, pred)
                
                # 마지막 단어 
                else:
                    
                    # print(line)
                    if '#' in word:

                        word = word.strip('#').strip()
                        
                        if pred == 'O':
                            line = line + word + " "
                            
                        else:
                            pii_word += word
                            line = line + "[{}:{}] ".format(pii_word, pred)
                            
                    else:
                        if pred == 'O':
                            line = line + word + " "
                        else:
                            line = line + "[{}:{}] ".format(word, pred) 
                
            f.write("{}\n".format(line.strip()))

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)
