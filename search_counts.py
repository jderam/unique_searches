import pandas as pd
import Levenshtein as lv
import datetime


# default settings
d_file_name = 'data/sample_data.txt'
d_time_delta = datetime.timedelta(seconds=10)
d_lev_dist = 3


def build_dataset(file_name: str = d_file_name,
                  time_delta: datetime.timedelta = d_time_delta,
                  lev_dist: int = d_lev_dist
                 ) -> pd.DataFrame:
    """ Reads in a data file.
        Creates 3 important columns.
        Each indicates "True" if the search is determined to be distinct from
        the search on the previous row based on the criteria given.
        
            new_search_user (bool): User is different than user on previous row.
            new_search_time (bool): The amount of time that's elapsed is greater
                                    than time_delta.
            new_search_word (bool): Levenshtein distance between current search
                                    term and previous search term exceeds lev_dist.
    """
    df = pd.read_csv(file_name, sep='\t')
    df['search_ts'] = pd.to_datetime(df.Timestamp)
    df.sort_values(by=['UserId', 'search_ts'], inplace=True, ignore_index=True)
    df['tdelta'] = df.groupby('UserId').search_ts.diff()
    df['prev_search'] = df.SearchTerm.shift()
    df['lv_dist'] = df.apply(lambda row: lv.distance(str(row['SearchTerm']), str(row['prev_search'])), axis=1)

    df['new_search_user'] = pd.isna(df['tdelta'])
    df['new_search_time'] = df['tdelta'].gt(time_delta)
    df['new_search_word'] = df['lv_dist'].gt(lev_dist)
    df['new_composite'] = (df['new_search_user'] | df['new_search_time'] | df['new_search_word'])
    return df


def unique_searches(df: pd.DataFrame) -> int:
    return df['new_composite'].sum()


def autosugg_searches(df: pd.DataFrame) -> int:
    sugg_search = (df['new_composite'].eq(True) & df['new_composite'].shift().eq(False))
    num = sugg_search.sum()
    return num


if __name__ == '__main__':
    df = build_dataset()

    search_count = unique_searches(df)
    print(f"unique searches: {search_count}")

    autosugg_count = autosugg_searches(df)
    print(f"auto-suggest searches: {autosugg_count}")

    pct_auto = round(autosugg_count / search_count * 100, 2)
    print(f"percentage of searches using auto-suggest: {pct_auto}%")
