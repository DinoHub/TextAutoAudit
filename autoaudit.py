from clearml import Task, Dataset, Logger, StorageManager


Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
task = Task.init(project_name='incubation c4', task_name='auto-audit', output_uri="s3://experiment-logging/storage/")
task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
task.execute_remotely(queue_name="compute", exit_process=True)
logger = task.get_logger()

dataset = Dataset.get(dataset_name="c4_raw_clean", dataset_project="datasets/c4")
dataset_folder = dataset.get_local_copy()

model_path = StorageManager.download_folder("s3://experiment-logging/storage/all-mpnet-base-v2","modules/sentence_transformers")

import pyarrow.parquet as pq
import pandas as pd
import os
import numpy as np
import spacy
import re
from tqdm import tqdm
from spellchecker import SpellChecker
from transformers import AutoTokenizer
from modules.sentence_transformers import SentenceTransformer, util
from clearml import Task, Dataset, Logger

class AutoAudit:

    def __init__(self, offline):
        self.offline = offline
        self.nlp = spacy.load('modules/spacy/en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0')
        self.wp_tokenizer = AutoTokenizer.from_pretrained("modules/tokenizers/word-piece") #bert
        self.sp_tokenizer = AutoTokenizer.from_pretrained("modules/tokenizers/sentence-piece") #t5
        self.bpe_tokenizer = AutoTokenizer.from_pretrained("modules/tokenizers/byte-pair") #gpt2
        self.spell = SpellChecker()

    def gen_stats(self, text):
        doc = self.nlp(text)
        word_len = len([token.text for token in doc])
        sent_len = len([sent.text for sent in doc.sents])
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
        
if __name__ == "__main__":
    pq_table = pq.read_table(dataset_folder)
    pq_table = pq_table.to_pandas()

    audit = AutoAudit(offline=True)
    compare = True
    raw_text_list = pq_table['raw'].tolist()
    clean_text_list = pq_table['clean'].tolist()
    word_len_list = []
    sent_len_list = []
    wp_len_list = []
    sp_len_list = []
    bpe_len_list = []
    sim_list = []
    spell_diff_list = []

    for i, (raw, clean) in tqdm(enumerate(zip(raw_text_list,clean_text_list)), total = len(raw_text_list)):
        word_len, sent_len, wp_len, sp_len, bpe_len = audit.gen_stats(raw)
        word_len_list.append(word_len)
        sent_len_list.append(sent_len)
        wp_len_list.append(wp_len)
        sp_len_list.append(sp_len)
        bpe_len_list.append(bpe_len)
        if compare:
            sim_score, orig_misspelled, proc_misspelled = audit.compare_stats(raw, clean)
            sim_list.append(sim_score)
            spell_diff_list.append(abs(len(orig_misspelled)-len(proc_misspelled)))
            logger.report_histogram("Sementic Similarity Distribution", "raw vs clean", iteration=i, values=sim_list, xaxis="Score",yaxis="Count")
            logger.report_histogram("No. of Spelling Mistakes Difference Distribution", "raw vs clean", iteration=i, values=spell_diff_list, xaxis="Spelling Mistakes Diff",yaxis="Count")
        