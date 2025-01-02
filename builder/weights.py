from loguru import logger
from tqdm import tqdm
import pandas as pd


def get_top_tickers(rating, top, quantile):
    rating = rating.sort_values('return', ascending=not top)
    top_tickers = rating.index[: int(len(rating) * quantile)]
    return top_tickers



class TopQuantileEqual:
    def __init__(self, prices, all_rating, rebalance_dates, top_quantile, filter_value=None,
                 filter_currency=None, filter_column=None):
        if isinstance(prices, dict):
            self.base_dict = prices

        else:
            raise
        self.all_rating = all_rating
        self.rebalance_dates = rebalance_dates
        self.top_quantile = top_quantile
        self.filter_value = filter_value
        self.filter_currency = filter_currency
        self.filter_column = filter_column

    def build_weights(self, top):
        struct_rebalance = {}

        logger.info("Computing rebalance structure:")
        for rebalance_date in tqdm(self.rebalance_dates):
            rating = self.all_rating[rebalance_date.strftime("%Y.%m.%d")]
            if rating.empty:
                logger.warning(f"Rating empty for {rebalance_date}, skipping...")
                continue
            fx_rate = 1
            if self.filter_currency is not None:
                fx_rate = self.filter_currency[:rebalance_date][-1]
            if self.filter_value is not None:
                min_turnover = fx_rate * self.filter_value
                top_tickers = get_top_tickers(rating, top, self.top_quantile)
            else:
                top_tickers = get_top_tickers(rating, top, self.top_quantile)
            struct = pd.DataFrame(index=top_tickers)
            if len(top_tickers) == 0:
                logger.warning(f'Skip {rebalance_date}')
                continue
            struct['weight'] = 1 / len(top_tickers)
            struct_rebalance[rebalance_date] = struct

        return struct_rebalance