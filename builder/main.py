from data.data_loader import get_forecasted_data
from rebal_dates import index_dates, rebal_dates
from rater import Rater
from weights import TopQuantileEqual
from builder.port_builder import PortfolioWithFees
import pandas as pd
import yfinance as yf
from config import PATH
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# для эндпоинта
base_dict = get_forecasted_data()

ind_dates = index_dates(base_dict)
# для эндпоинта
rating_builder = Rater(prices=base_dict,
                        momentum_period=5
                        )
price_momentum_rating = rating_builder.momentum(rating_dates=ind_dates)
# для эндпоинта
rebal_dates = rebal_dates(ind_dates)

weight = TopQuantileEqual(base_dict, price_momentum_rating, rebal_dates,
           top_quantile=0.1,
           filter_value=None,
           filter_currency=None,
           filter_column=None
           )
# для эндпоинта
leaders_weights, losers_weights = weight.build_weights(top=True), weight.build_weights(top=False)

ruonia_sample = pd.Series(index=ind_dates, dtype='float64')
ruonia_sample = ruonia_sample.fillna(0.00001)
leaders_pf = PortfolioWithFees(ind_dates, base_dict, leaders_weights, rebalance_for_days=1,
                       slippage=0.000001, multiplicator_rate=ruonia_sample)
leaders_pf.build_portfolio()
# для эндпоинта
leaders = pd.Series(leaders_pf.index_bars["close"], name='Close').to_frame()
# для эндпоинта
benchmark = yf.download("^BVSP", start=leaders.index[0], end=leaders.index[-1])
benchmark = benchmark.droplevel(1, axis=1)


overal_data = pd.concat([leaders, benchmark['Close']], axis = 1)
overal_data.columns = ['portfolio', 'index']
# для эндпоинта
cumprod = (overal_data.pct_change().dropna() + 1).cumprod() - 1




