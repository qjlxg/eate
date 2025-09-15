import pandas as pd
import akshare as ak
from datetime import datetime

class FundAnalyzer:
    """
    一个用于综合分析基金投资的工具类。
    能实时获取基金数据、基金经理数据并做出投资决策。
    """
    def __init__(self, fund_code: str, cache_data: bool = True, risk_free_rate: float = 0.02):
        self.fund_code = fund_code
        self.fund_data = {}
        self.market_data = {}
        self.manager_data = {}
        self.analysis_report = []
        self.cache_data = cache_data
        self.risk_free_rate = risk_free_rate
        self.cache = {}

    def _log(self, message: str):
        print(message)
        self.analysis_report.append(message)

    def get_real_time_fund_data(self):
        if self.cache_data and self.fund_code in self.cache.get('fund', {}):
            self._log(f"使用缓存数据 for 基金 {self.fund_code}")
            self.fund_data = self.cache['fund'][self.fund_code]
            return True

        self._log(f"正在获取基金 {self.fund_code} 的实时数据...")
        try:
            fund_data = ak.fund_open_fund_info_em(symbol=self.fund_code, indicator="单位净值走势")
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            fund_data.set_index('净值日期', inplace=True)
            
            returns = fund_data['单位净值'].pct_change().dropna()
            annual_returns = returns.mean() * 252
            annual_volatility = returns.std() * (252**0.5)
            sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
            
            rolling_max = fund_data['单位净值'].cummax()
            daily_drawdown = (fund_data['单位净值'] - rolling_max) / rolling_max
            max_drawdown = daily_drawdown.min() * -1
            
            self.fund_data = {
                'fund_code': self.fund_code,
                'latest_nav': fund_data['单位净值'].iloc[-1],
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            if self.cache_data:
                self.cache.setdefault('fund', {})[self.fund_code] = self.fund_data
            self._log(f"基金 {self.fund_code} 数据已获取：{self.fund_data}")
            return True
        except Exception as e:
            self._log(f"获取基金数据失败: {e}")
            return False

    def get_market_sentiment(self):
        self._log("正在获取市场情绪数据...")
        try:
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            last_week_data = index_data.iloc[-7:]
            
            price_change = last_week_data['close'].iloc[-1] / last_week_data['close'].iloc[0] - 1
            volume_change = last_week_data['volume'].mean() / last_week_data['volume'].iloc[:-1].mean() - 1
            if price_change > 0 and volume_change > 0:
                sentiment, trend = 'optimistic', 'bullish'
            elif price_change < 0:
                sentiment, trend = 'pessimistic', 'bearish'
            else:
                sentiment, trend = 'neutral', 'neutral'
            
            self.market_data = {'sentiment': sentiment, 'trend': trend}
            self._log(f"市场情绪数据已获取：{self.market_data}")
            return True
        except Exception as e:
            self._log(f"获取市场数据失败: {e}")
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown'}
            return False

    def get_fund_manager_data(self):
        self._log("正在获取基金经理数据...")
        try:
            manager_info = ak.fund_em_manager_fund_list(fund_code=self.fund_code)
            if manager_info.empty:
                self._log("未找到基金经理数据。")
                return False

            manager_info = manager_info.sort_values(by='上任日期', ascending=False).iloc[0]
            cumulative_return = float(manager_info['任职总回报'].replace('%', '')) if isinstance(manager_info['任职总回报'], str) else manager_info['任职总回报']
            
            self.manager_data = {
                'name': manager_info['基金经理'],
                'tenure_years': manager_info['任职天数'] / 365.0,
                'cumulative_return': cumulative_return
            }
            self._log(f"基金经理数据已获取：{self.manager_data}")
            return True
        except Exception as e:
            self._log(f"获取基金经理数据失败: {e}")
            return False

    def make_decision(self, personal_strategy: dict) -> str:
        self._log("-" * 25)
        self._log("开始做出投资决策:")

        if not self.fund_data or not self.market_data:
            return "数据获取不完整，无法给出明确建议。"

        market_trend = self.market_data.get('trend', 'unknown')
        fund_drawdown = self.fund_data.get('max_drawdown', float('inf'))
        invest_horizon = personal_strategy.get('horizon', 'unknown')
        risk_tolerance = personal_strategy.get('risk_tolerance', 'medium')
        sharpe_ratio = self.fund_data.get('sharpe_ratio', 0)

        manager_trust = False
        if self.manager_data:
            tenure_years = self.manager_data.get('tenure_years', 0)
            manager_return = self.manager_data.get('cumulative_return', -1)
            if tenure_years > 5 or manager_return > 20:
                manager_trust = True
                self._log(f"基金经理任职 {tenure_years:.2f} 年，累计回报 {manager_return:.2f}%，管理能力较强。")

        if invest_horizon == 'long-term':
            if market_trend == 'bearish':
                if fund_drawdown <= 0.2 and (risk_tolerance in ['medium', 'high'] or manager_trust):
                    return f"市场处于熊市，但基金回撤控制在 {fund_drawdown:.2f}，且经理能力较强。建议分批买入。"
                else:
                    return f"市场下跌，基金回撤 {fund_drawdown:.2f}，风险较高。建议观望或选择更稳健的基金。"
            else:
                if (sharpe_ratio > 1.5 or manager_trust) and risk_tolerance != 'low':
                    return f"市场处于牛市，基金夏普比率 {sharpe_ratio:.2f}，经理累计回报 {manager_return:.2f}%。适合继续持有或加仓。"
                else:
                    return f"市场上涨，但基金表现一般（夏普比率 {sharpe_ratio:.2f}）或风险偏好较低。建议评估其他基金。"

        elif invest_horizon == 'short-term':
            if sharpe_ratio > 1.5 and market_trend == 'bullish' and risk_tolerance != 'low':
                return f"市场处于牛市，基金夏普比率 {sharpe_ratio:.2f}，适合短期投资（风险承受能力 {risk_tolerance}）。建议适量买入。"
            else:
                return f"市场或基金指标不适合短期投资（风险承受能力 {risk_tolerance}）。建议保持谨慎。"

        return "投资策略与市场状况不匹配，请重新审视。"

if __name__ == '__main__':
    FUND_CODE = "005827"
    
    analyzer = FundAnalyzer(fund_code=FUND_CODE, risk_free_rate=0.03)
    analyzer.get_real_time_fund_data()
    analyzer.get_market_sentiment()
    analyzer.get_fund_manager_data()
    
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }

    decision = analyzer.make_decision(my_personal_strategy)

    print("\n--- 完整的基金投资分析报告 ---")
    print(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"分析基金: {FUND_CODE}")
    if analyzer.manager_data:
        print(f"基金经理: {analyzer.manager_data.get('name', 'N/A')}")
        print(f"任职年限: {analyzer.manager_data.get('tenure_years', 'N/A'):.2f} 年")
        print(f"任职总回报: {analyzer.manager_data.get('cumulative_return', 'N/A'):.2f}%")
    else:
        print("基金经理: 数据不可用")
        print("任职年限: 数据不可用")
        print("任职总回报: 数据不可用")
    print(f"当前市场趋势: {analyzer.market_data.get('trend')}")
    print(f"基金夏普比率: {analyzer.fund_data.get('sharpe_ratio', 0):.2f}")
    print(f"基金最大回撤: {analyzer.fund_data.get('max_drawdown', 0):.2f}")
    print(f"风险承受能力: {my_personal_strategy['risk_tolerance']}")
    print("-" * 25)
    print("投资决策:")
    print(decision)
    print("-" * 25)
