#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

import abc
import math
from six import with_metaclass, iteritems

import numpy as np
from pandas import isnull

from zipline.finance.transaction import create_transaction
from zipline.utils.cache import ExpiringCache

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3

SQRT_252 = math.sqrt(252)

DEFAULT_VOLUME_SLIPPAGE_BAR_LIMIT = 0.025
ROOT_SYMBOL_TO_ETA = {
    'AD': 0.046928040130566687,  # AUD
    'AI': 0.046928040130566687,  # Bloomberg Commodity Index
    'BD': 0.046928040130566687,  # Big Dow
    'BO': 0.046928040130566687,  # Soybean Oil
    'BP': 0.04784266952963729,   # GBP
    'CD': 0.051117234150758695,  # CAD
    'CL': 0.046928040130566687,  # Crude Oil
    'CM': 0.046928040130566687,  # Corn e-mini
    'CN': 0.046928040130566687,  # Corn
    'DJ': 0.023125339036681979,  # Dow Jones
    'EC': 0.048850768744160389,  # Euro FX
    'ED': 0.046928040130566687,  # Eurodollar
    'EE': 0.048714252127543864,  # Euro FX e-mini
    'EI': 0.031712708439692663,  # MSCI Emerging Markets mini
    'EL': 0.044207422018209361,  # Eurodollar NYSE LIFFE
    'ER': 0.045367444234542968,  # Russell2000 e-mini
    'ES': 0.04730613625075674,   # SP500 e-mini
    'ET': 0.046928040130566687,  # Ethanol
    'EU': 0.049750498465030997,  # Eurodollar e-micro
    'FC': 0.046928040130566687,  # Feeder Cattle
    'FF': 0.046928040130566687,  # 3-Day Federal Funds
    'FI': 0.046928040130566687,  # Deliverable Interest Rate Swap 5y
    'FS': 0.034557788010453824,  # Interest Rate Swap 5y
    'FV': 0.046928040130566687,  # US 5y
    'GC': 0.04893759042642172,   # Gold
    'HG': 0.052238437060378763,  # Copper
    'HO': 0.045122791828174927,  # Heating Oil
    'HU': 0.017154313062463938,  # Unleaded Gasoline
    'JE': 0.013948949613401812,  # JPY e-mini
    'JY': 0.046928040130566687,  # JPY
    'LB': 0.046928040130566687,  # Lumber
    'LC': 0.046928040130566687,  # Live Cattle
    'LH': 0.046928040130566687,  # Lean Hogs
    'MB': 0.046928040130566687,  # Municipal Bonds
    'MD': 0.046928040130566687,  # SP400 Midcap
    'ME': 0.030339483847094631,  # MXN
    'MG': 0.029579261656151684,  # MSCI EAFE mini
    'MI': 0.046928040130566687,  # SP400 Midcap e-mini
    'MS': 0.046928040130566687,  # Soybean e-mini
    'MW': 0.046928040130566687,  # Wheat e-mini
    'ND': 0.046928040130566687,  # Nasdaq100
    'NG': 0.047892536982022171,  # Natural Gas
    'NK': 0.046928040130566687,  # Nikkei225
    'NQ': 0.044770801723909417,  # Nasdaq100 e-mini
    'NZ': 0.049165039348092403,  # NZD
    'OA': 0.046928040130566687,  # Oats
    'PA': 0.046928040130566687,  # Palladium
    'PB': 0.046928040130566687,  # Pork Bellies
    'PL': 0.05458400761791686,   # Platinum
    'QG': 0.046928040130566687,  # Natural Gas e-mini
    'QM': 0.046928040130566687,  # Crude Oil e-mini
    'RM': 0.037413974839673583,  # Russell1000 e-mini
    'RR': 0.046928040130566687,  # Rough Rice
    'SB': 0.046928040130566687,  # Sugar
    'SF': 0.047788879925389501,  # CHF
    'SM': 0.046928040130566687,  # Soybean Meal
    'SP': 0.046928040130566687,  # SP500
    'SV': 0.05269381922380148,   # Silver
    'SY': 0.046928040130566687,  # Soybean
    'TB': 0.046928040130566687,  # Treasury Bills
    'TN': 0.046928040130566687,  # Deliverable Interest Rate Swap 10y
    'TS': 0.032908878455069152,  # Interest Rate Swap 10y
    'TU': 0.046928040130566687,  # US 2y
    'TY': 0.046928040130566687,  # US 10y
    'UB': 0.046928040130566687,  # Ultra Tbond
    'US': 0.046928040130566687,  # US 30y
    'VX': 0.046928040130566687,  # VIX
    'WC': 0.046928040130566687,  # Wheat
    'XB': 0.044451098475429676,  # RBOB Gasoline
    'XG': 0.046928040130566687,  # Gold e-mini
    'YM': 0.046928040130566687,  # Dow Jones e-mini
    'YS': 0.046928040130566687,  # Silver e-mini
}


class LiquidityExceeded(Exception):
    pass


def fill_price_worse_than_limit_price(fill_price, order):
    """
    Checks whether the fill price is worse than the order's limit price.

    Parameters
    ----------
    fill_price: float
        The price to check.

    order: zipline.finance.order.Order
        The order whose limit price to check.

    Returns
    -------
    bool: Whether the fill price is above the limit price (for a buy) or below
    the limit price (for a sell).
    """
    if order.limit:
        # this is tricky! if an order with a limit price has reached
        # the limit price, we will try to fill the order. do not fill
        # these shares if the impacted price is worse than the limit
        # price. return early to avoid creating the transaction.

        # buy order is worse if the impacted price is greater than
        # the limit price. sell order is worse if the impacted price
        # is less than the limit price
        if (order.direction > 0 and fill_price > order.limit) or \
                (order.direction < 0 and fill_price < order.limit):
            return True

    return False


class SlippageModel(with_metaclass(abc.ABCMeta)):
    """Abstract interface for defining a slippage model.
    """
    def __init__(self):
        self._volume_for_bar = 0

    @property
    def volume_for_bar(self):
        return self._volume_for_bar

    @abc.abstractproperty
    def process_order(self, data, order):
        """Process how orders get filled.

        Parameters
        ----------
        data : BarData
            The data for the given bar.
        order : Order
            The order to simulate.

        Returns
        -------
        execution_price : float
            The price to execute the trade at.
        execution_volume : int
            The number of shares that could be filled. This may not be all
            the shares ordered in which case the order will be filled over
            multiple bars.
        """
        pass

    def simulate(self, data, asset, orders_for_asset):
        self._volume_for_bar = 0
        volume = data.current(asset, "volume")

        if volume == 0:
            return

        # can use the close price, since we verified there's volume in this
        # bar.
        price = data.current(asset, "close")

        # BEGIN
        #
        # Remove this block after fixing data to ensure volume always has
        # corresponding price.
        if isnull(price):
            return
        # END
        dt = data.current_dt

        for order in orders_for_asset:
            if order.open_amount == 0:
                continue

            order.check_triggers(price, dt)
            if not order.triggered:
                continue

            txn = None

            try:
                execution_price, execution_volume = \
                    self.process_order(data, order)

                if execution_price is not None:
                    txn = create_transaction(
                        order,
                        data.current_dt,
                        execution_price,
                        execution_volume
                    )

            except LiquidityExceeded:
                break

            if txn:
                self._volume_for_bar += abs(txn.amount)
                yield order, txn

    def __eq__(self, other):
        return self.asdict() == other.asdict()

    def __hash__(self):
        return hash((
            type(self),
            tuple(sorted(iteritems(self.asdict())))
        ))

    def asdict(self):
        return self.__dict__


class EquitySlippageModel(SlippageModel):
    pass


class FutureSlippageModel(SlippageModel):
    pass


class VolumeSlippage(object):
    """
    Mixin for Equity and Future slippage models calculated from trade volume.
    """

    def __init__(self, volume_limit=DEFAULT_VOLUME_SLIPPAGE_BAR_LIMIT,
                 price_impact=0.1):

        self.volume_limit = volume_limit
        self.price_impact = price_impact

        super(VolumeSlippage, self).__init__()

    def __repr__(self):
        return """
{class_name}(
    volume_limit={volume_limit},
    price_impact={price_impact})
""".strip().format(class_name=self.__class__.__name__,
                   volume_limit=self.volume_limit,
                   price_impact=self.price_impact)

    def process_order(self, data, order):
        volume = data.current(order.asset, "volume")

        max_volume = self.volume_limit * volume

        # price impact accounts for the total volume of transactions
        # created against the current minute bar
        remaining_volume = max_volume - self.volume_for_bar
        if remaining_volume < 1:
            # we can't fill any more transactions
            raise LiquidityExceeded()

        # the current order amount will be the min of the
        # volume available in the bar or the open amount.
        cur_volume = int(min(remaining_volume, abs(order.open_amount)))

        if cur_volume < 1:
            return None, None

        # tally the current amount into our total amount ordered.
        # total amount will be used to calculate price impact
        total_volume = self.volume_for_bar + cur_volume

        volume_share = min(total_volume / volume,
                           self.volume_limit)

        price = data.current(order.asset, "close")

        # BEGIN
        #
        # Remove this block after fixing data to ensure volume always has
        # corresponding price.
        if isnull(price):
            return
        # END

        simulated_impact = volume_share ** 2 \
            * math.copysign(self.price_impact, order.direction) \
            * price
        impacted_price = price + simulated_impact

        if fill_price_worse_than_limit_price(impacted_price, order):
            return None, None

        return (
            impacted_price,
            math.copysign(cur_volume, order.direction)
        )


class VolumeShareSlippage(VolumeSlippage, EquitySlippageModel):
    """
    Model slippage as a function of the volume of shares traded.
    """
    pass


class VolumeContractSlippage(VolumeSlippage, FutureSlippageModel):
    """
    Model slippage as a function of the volume of contracts traded.
    """
    pass


class FixedSlippageBase(object):
    """
    Mixin class for Equity and Future slippage models calculated using a fixed
    spread.
    """

    def __init__(self, spread=0.0):
        self.spread = spread

    def process_order(self, data, order):
        price = data.current(order.asset, "close")

        return (
            price + (self.spread / 2.0 * order.direction),
            order.amount
        )


class FixedEquitySlippage(FixedSlippageBase, EquitySlippageModel):
    """
    Model slippage for equities as a fixed spread.

    Parameters
    ----------
    spread : float, optional
        spread / 2 will be added to buys and subtracted from sells.
    """
    pass


class FixedFutureSlippage(FixedSlippageBase, FutureSlippageModel):
    """
    Model slippage for futures as a fixed spread.

    Parameters
    ----------
    spread : float, optional
        spread / 2 will be added to buys and subtracted from sells.
    """
    pass


class WithWindowData(object):
    """
    Mixin class for slippage models requiring average daily volume and
    annualized daily volatility.
    """

    def __init__(self):
        super(WithWindowData, self).__init__()
        self._window_data_cache = ExpiringCache()

    def _get_window_data(self, data, asset, window_length):
        """
        Internal utility method to return the trailing 20-day mean volume and
        volatility of close prices for a specific asset.

        Parameters
        ----------
        data : The BarData from which to fetch the daily windows.
        asset : The Asset whose data we are fetching.

        Returns
        -------
        (mean volume, volatility)
        """
        try:
            values = self._window_data_cache.get(asset, data.current_session)
        except KeyError:
            # Add a day because we wan 'window_length' complete days.
            volume_history = data.history(
                asset, 'volume', window_length + 1, '1d',
            )
            close_history = data.history(
                asset, 'close', window_length + 1, '1d',
            )
            values = {
                'volume': volume_history[:-1].mean(),
                'close': close_history[:-1].pct_change().std() * SQRT_252,
            }
            self._window_data_cache.set(asset, values, data.current_session)

        return values['volume'], values['close']


class VolatilityVolumeShare(WithWindowData, FutureSlippageModel):

    def __init__(self, volume_limit):
        super(VolatilityVolumeShare, self).__init__()
        self.volume_limit = volume_limit

    def process_order(self, data, order):
        if order.open_amount == 0:
            return None, None

        minute_data = data.current(order.asset, ['volume', 'high', 'low'])
        volume = minute_data['volume']
        if not volume:
            return None, None

        # Price to use is the average of the minute bar's open and close.
        price = np.mean([minute_data['high'], minute_data['low']])

        eta = ROOT_SYMBOL_TO_ETA[order.asset.root_symbol]
        mean_volume, volatility = self._get_window_data(data, order.asset, 20)

        if mean_volume == 0:
            # If this is the first day the contract exists and there is no
            # volume history, default to an impact of 10 bps.
            simulated_impact = price * (10.0 / 10000)

            impacted_price = \
                price + math.copysign(simulated_impact, order.direction)
        else:
            txn_volume = int(
                min(volume * self.volume_limit, abs(order.open_amount))
            )
            psi = txn_volume / mean_volume

            market_impact = eta * volatility * math.sqrt(psi)

            # We divide by 10,000 because this model computes to basis points.
            # To convert from bps to % we need to divide by 100, then again to
            # convert from % to fraction.
            simulated_impact = (price * market_impact) / 10000

            impacted_price = \
                price + math.copysign(simulated_impact, order.direction)

        if fill_price_worse_than_limit_price(impacted_price, order):
            return None, None

        return impacted_price, math.copysign(txn_volume, order.direction)


# Alias FixedSlippage for backwards compatibility.
FixedSlippage = FixedEquitySlippage
