import pytest
import search_counts

df = search_counts.build_dataset()


def test_df_load():
    assert df.shape == (18, 11)


def test_search_count():
    u_searches = search_counts.unique_searches(df)
    assert u_searches == 5


def test_sugg_count():
    sugg_searches = search_counts.autosugg_searches(df)
    assert sugg_searches == 2

