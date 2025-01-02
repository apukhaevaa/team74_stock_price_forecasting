import pandas as pd

class Rater:
    def __init__(self, prices, momentum_period):
        self.momentum_period = momentum_period
        if isinstance(prices, dict):
            self.base_dict = prices
        else:
            raise

    def momentum(self, rating_dates):
        new_return = pd.concat(
            [
                ticker_data['close']
                .resample("1d")
                .last()
                .ffill()
                .pct_change(self.momentum_period)
                .reindex(rating_dates)
                .rename(ticker)
                for ticker, ticker_data in self.base_dict.items()
            ],
            axis=1,
        )
        new_return.sort_index(axis=1, inplace=True)

        all_rating = {
            dt.strftime("%Y.%m.%d"): pd.concat(
                [
                    new_return.loc[dt].rename('return'),
                ],
                axis=1,
                sort=True,
            ).dropna()
            for dt in rating_dates
        }

        return all_rating