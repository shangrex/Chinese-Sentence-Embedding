import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--model_name', type=str, required=False,
                    help='model\'s name', default='bert-base-chinese')

parser.add_argument('--txt', type=str, required=True,
                    help='the first sentence to be compared')

parser.add_argument('--txt2', type=str, required=True, 
                    help="the second sentence to be compared")

args = parser.parse_args()

txt = args.txt
txt2 = args.txt2

model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)

token = tokenizer.tokenize(txt)

seq_enc = tokenizer.convert_tokens_to_ids(token)

input_tensor = torch.tensor([seq_enc])

mask_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)

mask_tensor = mask_tensor.masked_fill(input_tensor != 0, 1)

output_seq = model(input_ids = input_tensor, attention_mask = mask_tensor)

output_tensor = output_seq[0]

# print(output_tensor)

token = tokenizer.tokenize(txt2)

seq_enc = tokenizer.convert_tokens_to_ids(token)

input_tensor = torch.tensor([seq_enc])

mask_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)

mask_tensor = mask_tensor.masked_fill(input_tensor != 0, 1)

output_seq = model(input_ids = input_tensor, attention_mask = mask_tensor)

output_tensor2 = output_seq[0]

# print(output_tensor2)

output = output_tensor.cpu().detach().numpy()[0]

output2 = output_tensor2.cpu().detach().numpy()[0]


output = np.sum(output, axis=0)

output2 = np.sum(output2, axis=0)


result = cosine_similarity([output], [output2])

print("similarity: ", result)
