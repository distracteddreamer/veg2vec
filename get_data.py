from multiprocessing import Process, Queue
import pandas as pd
import requests
import time
import os
import tqdm
import warnings
import argparse
import glob
from typing import List
from bs4 import BeautifulSoup as Soup
from utils import make_savepath


warnings.filterwarnings('ignore')

BASEPATH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DATABASE = 'pubmed'

def worker(task: Queue, done: Queue):
    while True:
        res = task.get() 
        if res == 'STOP':
            # This is safe because the main process adds 'STOP' for the first time
            # And does not add anything else after it. So nothing will be missed afterwards.
            task.put('STOP')
            break
        done.put(parse_result(res))

def parse_result(result):
    xml_soup = Soup(result)
    pubmedarticles = xml_soup.find_all('pubmedarticle')

    ids = []
    journals = []
    titles = []
    abstracts = []
    keywords = []
    year = []

    for pma in pubmedarticles:
        id_tag = pma.find('articleid', idtype='pubmed')
        if id_tag is None:
            continue
        ids.append(id_tag.text)

        abstract = pma.find_all('abstracttext')
        aa = ''
        if len(abstract) > 0:
            aa = ' '.join(at.text for at in abstract)
        abstracts.append(aa)

        title = pma.find('articletitle')
        tt = ''
        if title is not None:
            tt = title.text
        titles.append(tt)

        journal = pma.find('journal')
        jj = ''
        if journal is not None:
            jt = journal.find('title')
            if jt is not None:
                jj = jt.text
        journals.append(jj)

        kwlist = pma.find('keywordlist')
        kw = ''
        if kwlist is not None:
            kw = ','.join([k.text for k in kwlist.find_all('keyword')])

        keywords.append(kw)

        pmdate = pma.find_all('pubmedpubdate', pubstatus='medline')
        yr = -1
        if len(pmdate) > 0:
            yr = [p.find('year') for p in pmdate]
            yr = [int(y.text) for y in yr if y is not None]
            if len(yr):
                yr = min(yr)
        year.append(yr)

    return ids, journals, titles, abstracts, keywords, year

def get_query_params(result):
    xml_soup = Soup(result)
    count = xml_soup.find('count').text
    webenv = xml_soup.find('webenv').text
    query_key = xml_soup.find('querykey').text
    
    return int(count), webenv, query_key

def get_abstracts(term, num_records: int=500, parallel=True):

    params = dict(db=DATABASE, 
                term=term, 
                usehistory='y')
    req = requests.get(os.path.join(BASEPATH, 'esearch.fcgi'), params)
    print('Made a GET request to {}'.format(req.url))
    params = dict(rettype='xml', db=DATABASE)
    try:
        count, params['WebEnv'], params['query_key'] = get_query_params(req.text)
    except AttributeError as e:
        print ('The following error was raised: \n{}'.format(str(e)))
        return 
    print('{} records present for {} in {}'.format(count, term, DATABASE))

    if parallel:
        task_queue = Queue()
        done_queue = Queue()

        processes = []
        for p in range(os.cpu_count() - 1):
            processes.append(
                Process(target=worker, args=(task_queue, done_queue)))
            processes[-1].start()

    df = pd.DataFrame(columns=['pubmed_ids', 'journal', 'title', 'text', 'keywords',
    'rough_year'])

    iterrange = range(0, count, num_records)

    def _make_df(data):
        new_df = pd.DataFrame(
            dict(zip(df.columns, data))
        )
        return new_df.dropna().reset_index(drop=True)

    for start in tqdm.tqdm(iterrange):
        req = requests.get(os.path.join(BASEPATH, 'efetch.fcgi'), 
            dict(params, retstart=start, retmax=num_records))
        if parallel:
            task_queue.put(req.text)
        else:
            data = parse_result(req.text)
            df = df.append(_make_df(data)) 

        time.sleep(1)

    if parallel:
        task_queue.put('STOP')

        for _ in tqdm.tqdm(iterrange):  
            data = done_queue.get()
            df = df.append(_make_df(data)) 

        for p in processes:
            p.join()
    
    return count, df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--terms', required=False, default=None)
    parser.add_argument('--file', required=False, default=None)
    parser.add_argument('--savepath', required=False, default=None)
    
    args = parser.parse_args()

    if (args.terms is None) and (args.file is None):
        print("No terms to query. Either --terms or --file should have a value.")

    savepath = make_savepath(folder='results/data', savepath=args.savepath)

    if args.file is not None:
        with open(args.file) as f:
            terms = f.read().strip().split('\n')
    else:
        terms = args.terms.split(',')

    terms = [t for t in terms if len(t) > 0]

    df_info = pd.DataFrame(columns=['term', 'count', 'actual'])

    for term in terms:
        t = term.strip()
        print('Fetching results for {}'.format(t))
        abs_results = get_abstracts(t)
        if abs_results is None:
            df_info.append(dict(term=t, count=0, actual=0), ignore_index=True)
            continue
        count, df = abs_results
        df_info = df_info.append(dict(term=t, count=count, actual=len(df)), ignore_index=True)
        df.to_csv('{}/{}.csv'.format(savepath, t), index=False)

        # Save intermediate results so that if there is failure work is not wasted
        df_info.to_csv('{}/info.csv'.format(savepath))
