import pandas as pd
from loguru import logger
from tqdm import tqdm



class PortfolioWithFees:
    def __init__(self, index_dates, prices, weights, rebalance_for_days, slippage, multiplicator_rate,
                 max_weight=None):

        if max_weight:
            self.max_weight = max_weight
            self._normalize_weights()
        self.index_dates = index_dates
        if isinstance(prices, dict):
            self.base_dict = prices
        else:
            raise
        if isinstance(prices, dict):
            self.composition = weights.copy()
        else:
            raise
        self.rebalance_for_days = rebalance_for_days
        self.slippage = slippage
        self.multiplicator_rate = multiplicator_rate

        self.struct_daily = {}
        self.index_returns = [1]
        self.index_turnovers = [0]
        self.index_bars = None

        self.portfolio_weights = pd.Series(dtype='float64')

    def build_portfolio(self):
        rebalance_dates = pd.DatetimeIndex(sorted(self.composition.keys()))
        prev_date = rebalance_dates[rebalance_dates >= self.index_dates[0]][0]
        struct = self.composition[prev_date]
        self.portfolio_weights.loc[prev_date] = struct["weight"].sum()
        rebalance_dates = rebalance_dates[rebalance_dates >= prev_date]
        self.index_dates = self.index_dates[self.index_dates >= prev_date]

        logger.info("Computing daily structure:")
        for i, index_date in enumerate(tqdm(self.index_dates[1:]), start=1):
            index_return, index_turnover = self.calculate_daily_performance(
                prev_date, index_date, struct
            )
            self.index_returns.append(index_return)
            self.index_turnovers.append(index_turnover)
            last_rebalance = rebalance_dates[rebalance_dates < index_date]
            if len(last_rebalance) == 0:
                continue
            else:
                last_rebalance = last_rebalance[-1]

            days_from_rebalance = sum(self.index_dates[:i] > last_rebalance)
            if days_from_rebalance < self.rebalance_for_days:
                self.rebalance_at_day(struct, self.composition[last_rebalance], index_date)

            self.struct_daily[index_date] = struct.copy()
            prev_date = index_date

        index_returns = pd.Series(self.index_returns, index=self.index_dates)
        index_turnovers = pd.Series(self.index_turnovers, index=self.index_dates)

        index_values = (1 + index_returns).cumprod() * 1e3
        self.index_bars = pd.concat(
            [
                index_values.rename("close"),
                index_turnovers.rename("turnover"),
            ],
            axis=1,
        )

    def calculate_daily_performance(self, prev_date, curr_date, struct):
        struct["return"] = 0
        struct["turnover"] = 0
        for ticker in struct.index:
            try:
                prev_price = self.base_dict[ticker]["close"][:prev_date][-1]
                curr_price = self.base_dict[ticker]["close"][:curr_date][-1]
                turnover = 0
            except IndexError:
                logger.warning(f"Ticker {ticker} doesn't has price in {prev_date} or {curr_date}")
                continue

            current_return = (curr_price / prev_price - 1)
            if current_return > 0:
                struct.loc[ticker, "return"] = current_return * (1 - self.slippage)
            else:
                struct.loc[ticker, "return"] = current_return * (1 + self.slippage)
            struct.loc[ticker, "turnover"] = turnover

        last_portfolio_weights = self.portfolio_weights.index[-1]
        current_multiplicator_fee = self.multiplicator_rate.loc[prev_date] * (
                self.portfolio_weights.loc[last_portfolio_weights] - 1) / self.portfolio_weights.loc[
                                        last_portfolio_weights]
        index_return = (struct["return"] * struct["weight"]).sum() * (1 - current_multiplicator_fee)
        index_turnover = (struct["turnover"] * struct["weight"]).sum()

        struct["weight"] *= (1 + struct["return"]) / (1 + index_return)
        struct.drop(["return", "turnover"], axis=1, inplace=True)

        return index_return, index_turnover

    def rebalance_at_day(self, struct, target_struct, current_dt):
        for ticker, target in target_struct.iterrows():
            if ticker in struct.index:
                weight_change = (target["weight"] - struct.at[ticker, "weight"]) * (1 / self.rebalance_for_days)
                struct.loc[ticker, "weight"] += weight_change
            else:
                struct.loc[ticker] = target * (1 / self.rebalance_for_days)

        for ticker in struct.index.difference(target_struct.index):
            weight_change = struct.at[ticker, "weight"] * (1 / self.rebalance_for_days)
            struct.loc[ticker, "weight"] -= weight_change

            if struct.loc[ticker, "weight"] < 1e-5:
                struct.drop(ticker, inplace=True)


        sum_of_weights = struct["weight"].sum()
        self.portfolio_weights.loc[current_dt] = sum_of_weights

        other_cols = target_struct.drop(["weight"], axis=1).columns
        struct[other_cols] = target_struct[other_cols].reindex(struct.index)