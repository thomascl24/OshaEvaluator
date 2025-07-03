import json
import pandas as pd
import numpy as np
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
from tqdm import tqdm

from torch.nn.modules import CrossEntropyLoss
from torch import nn
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, get_linear_schedule_with_warmup
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
from functools import partial

import requests
from pathlib import Path
from urllib.parse import urljoin
from dotenv import load_dotenv
import re
import xml.etree.ElementTree as et
from typing import List, Optional

class PromptConstruction:
  def __init__(self, k, threshold, chunked_steps, vector_client, vector_collection, embeddings):
    self.top_k = k
    self.threshold = threshold
    self.client = vector_client
    self.collection = vector_collection
    self.embeddings = embeddings

    self.chunked_steps = chunked_steps

  def doc_search(self, query):
    qdrant_vector_store = Qdrant(
        client=self.client,
        collection_name=self.collection,
        # Specify the embedding function to use
        embeddings=self.embeddings
    )

    similar_docs = qdrant_vector_store.similarity_search_with_score(query, k=self.top_k)
    filtered_docs = [
        doc for doc in similar_docs if doc[1] >= self.threshold
    ]
    return filtered_docs

  def get_doc_details(self, docs, row, doc_id, index):
    try:
      return {
          "hypothesis": docs[index][0].page_content,
          "premise": row['step_text'],
          "source": docs[index][0].metadata['source'],
          "product": row['category'],
          "id": doc_id,
          "score": docs[index][1]
      }
    except:
      return None

  def retrieve_similar_docs(self):
    jsonl = []

    # Wrap the iterrows() with tqdm for progress tracking
    for id, row in tqdm(self.chunked_steps[['category', 'step_text']].iterrows(), total=self.chunked_steps.shape[0], desc="Processing Rows"):
        query = f"What regulation is related most closely to the following text?\n{row['step_text']}"
        similar_docs = self.doc_search(query)

        doc_id = 0

        # Process and print results for first similar document
        doc = self.get_doc_details(similar_docs, row, doc_id, id)

        if doc:
          jsonl.append(doc)

    return jsonl

from typing import TextIO, Iterable
import pandas as pd, json

class ManualChunker:
    def __init__(self, lines: Iterable[str]):
        self.lines = lines

    def chunk_data(self) -> pd.DataFrame:
        data = []
        for line in self.lines:
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("Skipping bad JSON line:", e)

        records = []
        for manual in data:
            title    = manual.get("Title", "")
            category = manual.get("Category", "appliances")
            steps    = manual.get("Steps", [])

            chunk, words = [], 0
            for step in steps:
                txt = " ".join(
                    l.get("Text", "") for l in step.get("Lines", []) if isinstance(l, dict)
                )
                words += len(txt.split())
                chunk.append(
                    {
                        "text":  txt,
                        "tools": step.get("Toolbox", []),
                        "parts": step.get("Parts", []),
                        "verbs": step.get("Verbs", []),
                    }
                )
                if words >= 200:
                    records.append(
                        {
                            "title":    title,
                            "category": category,
                            "step_text": " ".join(c["text"] for c in chunk),
                            "tools":    [t for c in chunk for t in c["tools"]],
                            "parts":    [p for c in chunk for p in c["parts"]],
                            "verbs":    [v for c in chunk for v in c["verbs"]],
                        }
                    )
                    chunk, words = [], 0

        df = pd.DataFrame(records)
        return (
            df.drop_duplicates(subset=["step_text", "title", "category"])
              .reset_index(drop=True)[["title", "category", "step_text"]]
        )


class SNLIDataset(Dataset):

    def __init__(self, directory, prefix, bert_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        label_map = {"contradiction": 0, 'neutral': 1, "entailment": 2}
        with open(os.path.join(directory, 'snli_1.0_' + prefix + '.jsonl'), 'r', encoding='utf8') as f:
            lines = f.readlines()
        self.result = []
        for line in lines:
            line_json = json.loads(line)
            if line_json['gold_label'] not in label_map:
                # print(line_json['gold_label'])
                continue
            self.result.append((line_json['sentence1'], line_json['sentence2'], label_map[line_json['gold_label']]))
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        sentence_1, sentence_2, label = self.result[idx]
        # remove .
        if sentence_1.endswith("."):
            sentence_1 = sentence_1[:-1]
        if sentence_2.endswith("."):
            sentence_2 = sentence_2[:-1]
        sentence_1_input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=False)
        sentence_2_input_ids = self.tokenizer.encode(sentence_2, add_special_tokens=False)
        input_ids = sentence_1_input_ids + [2] + sentence_2_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([label])

        return input_ids, label, length

def collate_to_max_length(batch, max_len, fill_values):
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    lengths = np.array([[len(field_data) for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * num_fields
    # [num_fields]
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = [torch.full([batch_size, max_lengths[field_idx]],
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields)]
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = batch[sample_idx][field_idx]
            output[field_idx][sample_idx][: data.shape[0]] = data
    # generate span_index and span_mask
    max_sentence_length = max_lengths[0]
    start_indexs = []
    end_indexs = []
    for i in range(1, max_sentence_length - 1):
        for j in range(i, max_sentence_length - 1):
            # if j - i > 10:
            #     continue
            start_indexs.append(i)
            end_indexs.append(j)
    # generate span mask
    span_masks = []
    for input_ids, label, length in batch:
        span_mask = []
        middle_index = input_ids.tolist().index(2)
        for start_index, end_index in zip(start_indexs, end_indexs):
            if 1 <= start_index <= length.item() - 2 and 1 <= end_index <= length.item() - 2 and (
                start_index > middle_index or end_index < middle_index):
                span_mask.append(0)
            else:
                span_mask.append(1e6)
        span_masks.append(span_mask)
    # add to output
    output.append(torch.LongTensor(start_indexs))
    output.append(torch.LongTensor(end_indexs))
    output.append(torch.LongTensor(span_masks))
    return output  # (input_ids, labels, length, start_indexs, end_indexs, span_masks)

class ExplainNLP(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.bert_dir = 'roberta-large'
    self.model = ExplainableModel(self.bert_dir)
    self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_dir)
    self.loss_fn = CrossEntropyLoss()
    self.train_acc = MulticlassAccuracy(num_classes=3)
    self.valid_acc = MulticlassAccuracy(num_classes=3)
    self.train_acc = MulticlassAccuracy(num_classes=3)
    self.args = {
        'weight_decay': 0.0,
        'lr': 2e-5,
        'adam_epsilon': 1e-9,
        'warmup_steps': 0,
        'max_epochs': 20,
        'accumulate_grad_batches': 1,
        'workers': 2,
        'lamb':1,
        'span_topk':5,
        'save_path':'./save_res',
        'data_dir': './snli_data',
        'batch_size': 32,
        'max_length': 128,
    }
    self.output = []
    self.check_data = []

    save_path = self.args.get('save_path')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

  def forward(self, input_ids, start_indexs, end_indexs, span_masks):
    return self.model(input_ids, start_indexs, end_indexs, span_masks)

  def configure_optimizers(self):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.args.get('weight_decay'),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98),  # according to RoBERTa paper
                      lr=self.args.get('lr'),
                      eps=self.args.get('adam_epsilon'))
    t_total = len(self.train_dataloader()) // self.args.get('accumulate_grad_batches') * self.args.get('max_epochs')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.get('warmup_steps'),
                                                num_training_steps=t_total)

    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

  def train_dataloader(self):
    return self.get_dataloader(prefix="train")

  def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        dataset = SNLIDataset(directory=self.args.get('data_dir'), prefix=prefix,
                                  bert_path=self.bert_dir,
                                  max_length=self.args.get('max_length'))
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.get('batch_size'),
            num_workers=self.args.get('workers'),
            collate_fn=partial(collate_to_max_length, max_len=None, fill_values=[1, 0, 0]),
            drop_last=False
        )
        return dataloader

  def compute_loss_and_acc(self, batch, mode='train'):
    input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
    y = labels.view(-1)
    y_hat, a_ij = self.forward(input_ids, start_indexs, end_indexs, span_masks)
    # compute loss
    ce_loss = self.loss_fn(y_hat, y)
    reg_loss = self.args.get('lamb') * a_ij.pow(2).sum(dim=1).mean()
    loss = ce_loss - reg_loss
    # compute acc
    predict_scores = F.softmax(y_hat, dim=1)
    predict_labels = torch.argmax(predict_scores, dim=-1)
    if mode == 'train':
        acc = self.train_acc(predict_labels, y)
    else:
        acc = self.valid_acc(predict_labels, y)
    # if test, save extract spans
    if mode == 'test':
        values, indices = torch.topk(a_ij, self.args['span_topk'])
        values = values.tolist()
        indices = indices.tolist()
        for i in range(len(values)):
            input_ids_list = input_ids[i].tolist()
            origin_sentence = self.tokenizer.decode(input_ids_list, skip_special_tokens=True)
            self.output.append(
                str(labels[i].item()) + '<->' + str(predict_labels[i].item()) + '<->' + origin_sentence + '\n')
            for j, span_idx in enumerate(indices[i]):
                score = values[i][j]
                start_index = start_indexs[span_idx]
                end_index = end_indexs[span_idx]
                pre = self.tokenizer.decode(input_ids_list[:start_index], skip_special_tokens=True)
                high_light = self.tokenizer.decode(input_ids_list[start_index:end_index + 1],
                                                    skip_special_tokens=True)
                post = self.tokenizer.decode(input_ids_list[end_index + 1:], skip_special_tokens=True)
                span_sentence = pre + '【' + high_light + '】' + post
                self.output.append(format('%.4f' % score) + "->" + span_sentence + '\n')
                if j == 0:
                    self.check_data.append(str(labels[i].item()) + '\t' + high_light + '\n')
            self.output.append('\n')

    return loss, acc

  def on_validation_epoch_end(self):
    # log epoch metric
    self.valid_acc.compute()
    self.log('valid_acc_end', self.valid_acc.compute())

  def training_step(self, batch, batch_idx):
    loss, acc = self.compute_loss_and_acc(batch)
    self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
    self.log('train_acc', acc, on_step=True, on_epoch=False)
    self.log('train_loss', loss)
    return loss

  def val_dataloader(self):
    return self.get_dataloader(prefix="dev")

  def validation_step(self, batch, batch_idx):
    loss, acc = self.compute_loss_and_acc(batch, mode='dev')
    self.log('valid_acc', acc, on_step=False, on_epoch=True)
    self.log('valid_loss', loss)
    return loss

  def test_dataloader(self):
    return self.get_dataloader(prefix="test")

  def test_step(self, batch, batch_idx):
    loss, acc = self.compute_loss_and_acc(batch, mode='test')
    self.log("test_loss", loss, on_step=False, on_epoch=True)
    self.log("test_acc", acc, on_step=False, on_epoch=True)
    return {'test_loss': loss, "test_acc": acc}


  def on_test_epoch_end(self):
      # Save explanation outputs to a file
      output_file_path = os.path.join(self.args['save_path'], 'explanation.txt')
      with open(output_file_path, 'w', encoding='utf-8') as f:
          f.writelines(self.output)

      # Save check_data (top-1 spans) for evaluation/inspection
      check_data_file_path = os.path.join(self.args['save_path'], 'check_data.txt')
      with open(check_data_file_path, 'w', encoding='utf-8') as f:
          f.writelines(self.check_data)

      # Log test accuracy to the logger
      test_outputs = self.trainer.callback_metrics
      if 'test_acc' in test_outputs:
          self.log('test_acc_end', test_outputs['test_acc'])

class ExplainableModel(nn.Module):
    def __init__(self, bert_dir):
        super().__init__()
        self.bert_config = RobertaConfig.from_pretrained(bert_dir, output_hidden_states=False, num_labels=3)
        self.intermediate = RobertaModel.from_pretrained(bert_dir, return_dict=False)
        self.span_info_collect = SICModel(self.bert_config.hidden_size)
        self.interpretation = InterpretationModel(self.bert_config.hidden_size)
        self.output = nn.Linear(self.bert_config.hidden_size, self.bert_config.num_labels)

    def forward(self, input_ids, start_indexs, end_indexs, span_masks):
        # generate mask
        attention_mask = (input_ids != 1).long()
        # intermediate layer
        hidden_states, first_token = self.intermediate(input_ids, attention_mask=attention_mask)  # output.shape = (bs, length, hidden_size)
        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(hidden_states, start_indexs, end_indexs)
        # interpretation layer
        H, a_ij = self.interpretation(h_ij, span_masks)
        # output layer
        out = self.output(H)
        return out, a_ij


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indexs, end_indexs):
        W1_h = self.W_1(hidden_states)  # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = torch.index_select(W1_h, 1, start_indexs)  # (bs, span_num, hidden_size)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indexs)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indexs)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indexs)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indexs)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indexs)

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        h_ij = torch.tanh(span)
        return h_ij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        o_ij = self.h_t(h_ij).squeeze(-1)  # (ba, span_num)
        # mask illegal span
        o_ij = o_ij - span_masks
        # normalize all a_ij, a_ij sum = 1
        a_ij = nn.functional.softmax(o_ij, dim=1)
        # weight average span representation to get H
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)  # (bs, hidden_size)
        return H, a_ij


model_name = "sentence-transformers/all-mpnet-base-v2"

# Initialize the HuggingFaceEmbeddings object
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    # Specify the directory where the model will be cached
    cache_folder="./models",
    # Set to True to ensure the model is loaded into GPU memory if available
    model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {'device': 'cpu'},
    # Set to True to ensure the model is loaded for inference
    encode_kwargs={'batch_size': 64}
)


class PremiseHypothesisDataset(Dataset):
    _label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}

    def __init__(self,
                 tokenizer,
                 max_length: int = 128,
                 top_k: int = 3,
                 threshold: int = 0,
                 chunked_steps: pd.DataFrame = None,
                 vector_client: Optional[QdrantClient]=None,
                 vector_collection: str = 'osha_regulations',
                 embedding = hf_embeddings):
        self.max_length = max_length
        self.tokenizer = tokenizer

        prompt_constructor = PromptConstruction(k=top_k,
                                                threshold=threshold,
                                                chunked_steps=chunked_steps,
                                                vector_client=vector_client,
                                                vector_collection=vector_collection,
                                                embeddings=embedding
                                                )
        self.samples = prompt_constructor.retrieve_similar_docs()

    def __len__(self):
        return len(self.samples)

    def _encode(self, s: str) -> List[int]:
        if s.endswith("."):
            s = s[:-1]
        return self.tokenizer.encode(s, add_special_tokens=False)

    def __getitem__(self, idx):
        row = self.samples[idx]
        premise_ids = self._encode(row["premise"])
        hypo_ids    = self._encode(row["hypothesis"])

        input_ids   = premise_ids + [2] + hypo_ids       # 2 == </s> in RoBERTa
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[: self.max_length - 2]

        # tensors expected by collate_to_max_length
        input_ids = torch.LongTensor([0] + input_ids + [2])   # prepend <s>, append </s>
        length    = torch.LongTensor([len(input_ids)])
        label_str = row.get("label", "contradiction")         # dummy if absent
        label_id  = self._label2id.get(label_str, 0)
        label     = torch.LongTensor([label_id])

        return input_ids, label, length
    
class Predictor:
  def __init__(self, model, out_path: str=None, batch_size: int=32, num_workers: int=2, device: str=None, chunked_steps: pd.DataFrame=None, vector_client: Optional[QdrantClient]=None):
    self.model = model
    self.out_path = out_path
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.device = device
    self.chunked_steps=chunked_steps
    self.client = vector_client

  def predict(self):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.model.eval()
      self.model.to(device)

      dataset   = PremiseHypothesisDataset(self.model.tokenizer,
                                          max_length=self.model.args["max_length"],
                                           chunked_steps=self.chunked_steps, vector_client=self.client)
      dataloader = DataLoader(dataset,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              shuffle=False,
                              collate_fn=partial(collate_to_max_length,
                                                max_len=None,
                                                fill_values=[1, 0, 0])
                              )


      id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
      predictions: List[str] = []

      # 2.3  Inference loop
      with torch.no_grad():
          for batch in tqdm(dataloader):
              # move everything to the same device
              batch = [x.to(device) for x in batch]
              input_ids, labels, length, start_idxs, end_idxs, span_masks = batch

              logits, _ = self.model(input_ids, start_idxs, end_idxs, span_masks)
              preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)  # [bs]

              predictions.extend(id2label[p.item()] for p in preds)

      return {'predictions': predictions, 
              'hypotheses': list([samp['hypothesis'] for samp in dataset.samples]), 
              'premises': list([samp['premise'] for samp in dataset.samples])} 
