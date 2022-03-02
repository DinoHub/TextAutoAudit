from clearml import Task, Dataset, Logger, StorageManager
import hydra
from omegaconf import OmegaConf
from model import AutoAudit        

@hydra.main(config_path='.', config_name="config")
def main(cfg):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    task = Task.init(project_name='incubation c4', task_name='auto-audit', output_uri="s3://experiment-logging/storage/")
    cfg_dict = task.connect_configuration(name='dictionary', configuration=cfg_dict)
    task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.execute_remotely(queue_name="compute", exit_process=True)
    logger = task.get_logger()

    dataset = Dataset.get(dataset_name="c4_raw_clean", dataset_project="datasets/c4")
    dataset_folder = dataset.get_local_copy()

    # preds = Dataset.get(dataset_name="c4_raw_clean", dataset_project="datasets/c4")
    # preds_folder = dataset.get_local_copy()


    model_path = StorageManager.download_folder("s3://experiment-logging/storage/all-mpnet-base-v2")

    import pyarrow.parquet as pq
    import pandas as pd
    from scipy import stats
    from tqdm import tqdm
    from clearml import Task, Dataset
    import plotly.express as px
    from model import AutoAudit 

    #args
    sample_size = cfg_dict["sample_size"]
    compare = cfg_dict["compare"]
    offline = cfg_dict["offline"]

    #main code
    pq_table = pq.read_table(dataset_folder)
    pq_table = pq_table.to_pandas()
    pq_table = pq_table.sample(n=sample_size)


    audit = AutoAudit(offline=offline)
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
        word_len, sent_len, wp_len, sp_len, bpe_len = audit.gen_stats(raw, batch_size = 30)
        word_len_list.append(word_len)
        sent_len_list.append(sent_len)
        wp_len_list.append(wp_len)
        sp_len_list.append(sp_len)
        bpe_len_list.append(bpe_len)
        stat_df = pd.DataFrame()
        stat_df['word_len'] = word_len_list
        stat_df['sent_len'] = sent_len_list
        stat_df['wp_len'] = wp_len_list
        stat_df['sp_len'] = sp_len_list
        stat_df['bpe_len'] = bpe_len_list
        fig_1 = px.histogram(stat_df, x='word_len')
        fig_2 = px.histogram(stat_df, x='sent_len')
        fig_3 = px.histogram(stat_df, x='wp_len')
        fig_4 = px.histogram(stat_df, x='sp_len')
        fig_5 = px.histogram(stat_df, x='bpe_len')
        logger.report_plotly(title="No. of Words Distribution", iteration=i, figure=fig_1, series="")
        logger.report_plotly(title="No. of Sentences Distribution", iteration=i, figure=fig_2, series="")
        logger.report_plotly(title="No. of WordPieces Tokens Distribution", iteration=i, figure=fig_3, series="")
        logger.report_plotly(title="No. of SentencePieces Tokens Distribution", iteration=i, figure=fig_4, series="")
        logger.report_plotly(title="No. of BPE Tokens Distribution", iteration=i, figure=fig_5, series="")
        if compare:
            sim_score, orig_misspelled, proc_misspelled = audit.compare_stats(raw, clean)
            sim_list.append(sim_score)
            spell_diff_list.append(abs(len(orig_misspelled)-len(proc_misspelled)))
            stat_df['semantic'] = sim_list
            stat_df['spelling'] = spell_diff_list
            fig_6 = px.histogram(stat_df, x='semantic')
            fig_7 = px.histogram(stat_df, x='spelling')
            logger.report_plotly(title="Sementic Similarity Distribution", iteration=i, figure=fig_6, series="")
            logger.report_plotly(title="No. of Spelling Mistakes Difference Distribution", iteration=i, figure=fig_7, series="")
    stat_df['doc_id'] = pq_table['doc_id']

if __name__ == "__main__":
    main()