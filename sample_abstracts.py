import pandas as pd
import argparse
import random
import numpy as np
import json

from collections import deque

from utils import make_savepath

def get_samples(num_samples, num_abstracts):
    """
        Generates up to num_sample indices uniformly
        across the dataframes and partions them

        Args:
            num_samples: max number of samples, 
                returns min(num_samples, sum(num_abstracts)) samples
            num_abstracts: array of number of abstacts per dataframe
        Returns:
            a list with indices corresponding to each dataframe in num_abstracts,
                some of which may be empty
    """
    shifts = np.concatenate([[0], np.cumsum(num_abstracts[:-1])])
    total_abstracts = np.sum(num_abstracts)
    num_samples_actual = min(num_samples, total_abstracts)

    choices = np.sort(np.random.choice(total_abstracts, num_samples_actual))
    split_inds = np.searchsorted(choices, shifts)[1:]
    splits = np.split(choices, split_inds)

    sample_inds = [split - shift for split, shift in zip(splits, shifts)]
    return sample_inds


parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=1000,
    help='Maximum number of samples (limited by number available)')
parser.add_argument('--num_per_term', type=int, required=False,
    help="Maximum number of samples per term - if provided then "
        "this many samples will be obtained for each term "
        "and from these num_samples will be selected")
parser.add_argument('--folders', nargs='+', type=int, required=True)
parser.add_argument('--exclude', nargs='+', type=str, required=False)
parser.add_argument('--include', nargs='+', type=str, required=False)
args = parser.parse_args()

def main():    
    dfs = []
    for f in args.folders:
        path = 'results/data_{}'.format(f)
        df = pd.read_csv('{}/info.csv'.format(path))
        df = df.assign(path = (path + '/' + df.term + '.csv'))
        dfs.append(df)

    print('Getting samples for folders {}'.format(
        ', '.join(map('data_{}'.format, (args.folders))) 
    ))

    df_info = pd.concat(dfs).reset_index()

    if args.exclude is not None:
        df_info = df_info[~df_info.term.isin(args.exclude)].reset_index()

    elif args.include is not None:
        df_info = df_info[df_info.term.isin(args.include)].reset_index()

    if args.num_per_term is None:
        sample_inds = get_samples(args.num_samples, df_info.actual.values)  

    assert set(df_info.index.values) == set(range(len(df_info)))

    samples = []

    for i, row in df_info.iterrows():
        df = pd.read_csv(row.path)
        assert len(df) == row.actual
        df = df.dropna()
        df = df.loc[np.all([df.text.str.contains(r) for r in row.term.split()], axis=0)]
        num_not_na = len(df)

        if (args.num_per_term is None):
            df = df.loc[sample_inds[i]]
        else:
            df = df.sample(min(args.num_per_term, len(df)), 
                            replace=False)
                            
        samples.append(df.assign(src=row.path, 
                                 term=row.term, 
                                 orig_index=df.index.values))
        
        print('{} samples from {} which has {} abstracts'.format(
            len(df), row.path, num_not_na
        ))

    df_samples = pd.concat(samples).drop_duplicates(
        ['pubmed_ids', 'journal', 'title', 'text']
    ).reset_index()

    if len(df_samples) > args.num_samples:
        df_samples = df_samples.sample(args.num_samples, replace=False).reset_index()

    savepath = make_savepath(folder='samples')

    df_path  = '{}/abstracts.csv'.format(savepath)
    df_samples.to_csv(df_path, index=False)

    print('Sampled {} abstracts, saved to {}'.format(len(df_samples), df_path))

    with open('{}/args.json'.format(savepath), 'w') as f:
        json.dump(fp=f, obj=args.__dict__)

if __name__ == '__main__':
    main()