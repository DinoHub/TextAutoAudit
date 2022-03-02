import sys
sys.path.append('modules/sentence-transformers')
import pyarrow.parquet as pq
import pandas as pd
from scipy import stats
import os
import numpy as np
import spacy
import re
from tqdm import tqdm
from spellchecker import SpellChecker
from transformers import BertTokenizerFast, T5TokenizerFast, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util
from clearml import Task, Dataset, Logger
import plotly.express as px
import itertools
from omegaconf import OmegaConf
import hydra


class AutoAudit:

    def __init__(self, offline):
        self.offline = offline
        self.nlp = spacy.load('modules/spacy/en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0')
        self.wp_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") #bert
        self.sp_tokenizer = T5TokenizerFast.from_pretrained("t5-small") #t5
        self.bpe_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") #gpt2
        self.spell = SpellChecker()

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def gen_stats(self, text, batch_size):
        doc = self.nlp(text)
        word_len = len([token.text for token in doc])
        sents = [sent.text for sent in doc.sents]
        sent_len = len(sents)
        if batch_size>0:
            batch_sents = self.chunks(sents,batch_size)
            wp_len_ls = []
            sp_len_ls = []
            bpe_len_ls = []
            for batch in batch_sents:
                wp_len_ls += list(itertools.chain.from_iterable(self.wp_tokenizer(batch, truncation=False)['input_ids']))
                sp_len_ls += list(itertools.chain.from_iterable(self.sp_tokenizer(batch, truncation=False)['input_ids']))
                bpe_len_ls += list(itertools.chain.from_iterable(self.bpe_tokenizer(batch, truncation=False)['input_ids']))
            wp_len = len(wp_len_ls)
            sp_len = len(sp_len_ls)
            bpe_len = len(bpe_len_ls)
        else:
            wp_len = len(self.wp_tokenizer(text, truncation=False)['input_ids'])
            sp_len = len(self.sp_tokenizer(text, truncation=False)['input_ids'])
            bpe_len = len(self.bpe_tokenizer(text, truncation=False)['input_ids'])
        return word_len, sent_len, wp_len, sp_len, bpe_len

    def sent_proc(self, model, text):
        doc = self.nlp(text)
        sentiment_list = []
        for sent in doc.sents:
            sentiment_list.append(model.encode(sent.text))
        return sentiment_list


    def compare_stats(self, orig_text, proc_text):
        model = SentenceTransformer(model_path)
        orig_sentiment_list = self.sent_proc(model, orig_text)
        proc_sentiment_list = self.sent_proc(model, proc_text)
        orig_doc = np.mean(orig_sentiment_list, axis=0)
        proc_doc = np.mean(proc_sentiment_list, axis=0)
        sim_score = util.cos_sim(orig_doc,proc_doc).item()
        orig_misspelled = self.spell.unknown(re.sub(r"[,.;@#?!&$]+\ *"," ",orig_text).split())
        proc_misspelled = self.spell.unknown(re.sub(r"[,.;@#?!&$]+\ *"," ",proc_text).split())
        return sim_score, orig_misspelled, proc_misspelled