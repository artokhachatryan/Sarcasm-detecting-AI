!pip install transformers
!pip install sentencepiece
!pip install torch
 
 
import nltk
import nltk.corpus
import json
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
 
 
 
 
 
def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)
data = list(parse_data('./Sarcasm_Headlines_Dataset_v2.json'))
headline_list  = []
sarcasm_check = []
for item in data:
    headline_list.append(item["headline"])
    sarcasm_check.append(item["is_sarcastic"])
df =  pd.DataFrame(data)
df["modul_truth_values"] = df["is_sarcastic"].apply(lambda input: [0,1] if  input  == 1 else [1,0])
print(df["modul_truth_values"])


# ---------------------------------------------------------------------------------------


tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
inputs = tokenizer("Hello World!", return_tensors="pt")
outputs = model(**inputs)

# ---------------------------------------------------------------------------------------


def get_BERT_input(headlines, tokenizer):
  input_ids = []
  attention_masks = []
  encoded_dict = tokenizer.batch_encode_plus(
                      headlines,                     
                      add_special_tokens = True, 
                      max_length = 65,           
                      pad_to_max_length = True,
                      return_attention_mask = True,   # Construct attn. masks.
                      return_tensors = 'pt',     # Return pytorch tensors.
                )
  input_ids, attention_mask = encoded_dict['input_ids'], encoded_dict['attention_mask']
  # Add the encoded sentence to the list.    
  input_ids = encoded_dict['input_ids']
  
  # And its attention mask (simply differentiates padding from non-padding).
  attention_mask = encoded_dict['attention_mask']
  return torch.tensor(input_ids), torch.tensor(attention_mask)
  
class SarcasmDataset(Dataset):
    def __init__(self, df):
        self.headlines = df["headline"]
        self.modul_truth_values = df["modul_truth_values"]
    def __len__(self):
        return len(self.headlines)
 
    def __getitem__(self, idx):
        return self.headlines.iloc[idx],torch.tensor(self.modul_truth_values.iloc[idx])
    
 
#get_BERT_input(["Shahbazis wearing a plaid shirt", "We have 60. students in this class"],tokenizer)
 
 
dataset = SarcasmDataset(df)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



#---------------------------------------------------------------------------------------


from torch.optim import AdamW
class SarcasmModel(nn.Module):
    def __init__(self):
        super(SarcasmModel, self).__init__()
        self.XLM = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.hidden_1 = nn.Linear(768, 2)
        
        self.to_delete = 2
        self.softmax  = nn.Softmax(dim=1)
 
    def forward(self, b_input_ids,b_attention_mask):
        bert_output = self.XLM(b_input_ids,b_attention_mask)
        hidden_state = bert_output["last_hidden_state"]
        sentence_vector = torch.mean(hidden_state,dim = 1)
       # print(sentence_vector.shape)
        x = self.hidden_1(sentence_vector)
       # print(x.shape)
       
        probabilties = self.softmax(x)
        #print(probabilties)
        
 
        # print(bert_output)
        # print(b_input_ids.shape)
        # print(b_attention_mask.shape)
        return probabilties
 
model = SarcasmModel()
loss_function = nn.BCELoss().to(device)
optimizer = AdamW(model.parameters(),
                  lr = 2e-5 ,
                  eps = 1e-8
                  )
model = model.to(device)
for i_batch,(b_headlines,b_modul_truth_values) in enumerate(train_dataloader):
  b_input_ids, b_attention_mask = get_BERT_input(b_headlines,tokenizer)
  # print(b_input_ids.shape)
  # print(b_attention_mask.shape)
  optimizer.zero_grad()
  b_prediciton = model(b_input_ids.to(device),b_attention_mask.to(device))
  loss = loss_function(b_prediciton,b_modul_truth_values.float().to(device))
  #print(loss)
  model.zero_grad()
  loss.backward()
  optimizer.step()
  # print(loss)
  batch_size = 32
  if i_batch % 10 == 0:
      iteration = i_batch*batch_size
      print("Iteration:", i_batch*batch_size, "Loss:", loss.data)
      batch_accuracy = torch.mean(torch.sum(b_prediciton * b_modul_truth_values.to(device), dim=1))
      print("Batch Accuracy:", batch_accuracy.data*100)
      if iteration % 5120 == 0:
        # torch.save(model.state_dict(), expt_folder + "SarcasmModel.pt")
        print("Saved Model")


#---------------------------------------------------------------------------

def predict(b_headline, tokenizer, model):
  b_input_ids, b_attention_mask = get_BERT_input(b_headline, tokenizer)

  b_predictions = model(b_input_ids.to(device), b_attention_mask.to(device))
  print(b_predictions)
  sarcastic_probability = b_predictions.data[0][1].item() * 100
  not_sarcastic_probability = b_predictions.data[0][0].item() * 100
  print_string = "Sarcastic:", f'{sarcastic_probability:.2f}', "Not Sarcastic:", f'{not_sarcastic_probability:.2f}'
  # return b_predictions
  return print_string



predict(["Trump Forced To Shut Down Blog After Publishing Hulk Hogan Sex Tape"], tokenizer, model)


