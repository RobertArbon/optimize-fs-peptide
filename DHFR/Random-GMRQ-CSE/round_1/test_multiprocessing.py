import multiprocessing as mp
import pandas as pd
import numpy as np



def f(config):
    idx = config['row']
    result = [config['seed']]*3
    result2 = [config['seed']*3]*3

    return {'row' :idx, 'score': result, 'score2': result2}


if __name__ == "__main__":

    configs = [{'row': 1, 'seed': 10},
               {'row': 2, 'seed': 9},
               {'row': 3, 'seed': 8},
               {'row': 4, 'seed': 7}]

    # df = pd.

    with mp.Pool(4) as pool:
        all_results = list(pool.imap_unordered(f, configs))

    print(all_results)
    index = [x['row'] for x in all_results]
    data = {'scores': [x['score'] for x in all_results],
            'scores2': [x['score2'] for x in all_results]}
    df2 = pd.DataFrame(data, index=index)
    print(df2)
