import pandas as pd
import akshare as ak
from datetime import datetime

class FundAnalyzer:
    """
    一个用于综合分析基金投资的工具类。
    能实时获取基金数据并做出投资决策。
    """
    def __init__(self, fund_code: str):
        self.fund_code = fund_code
        self.fund_data = {}
        self.manager_data = {}
        self.market_data = {}

    def get_real_time_fund_data(self):
        """
        使用 akshare 库获取基金的实时数据。
        包括最新净值、夏普比率和最大回撤。
        """
        print(f"正在获取基金 {self.fund_code} 的实时数据...")
        try:
            # 获取基金历史净值数据，用于计算指标
            fund_data = ak.fund_open_fund_info_em(fund_code=self.fund_code, indicator="单位净值走势")
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            fund_data.set_index('净值日期', inplace=True)
            
            # 简化版指标计算
            returns = fund_data['单位净值'].pct_change().dropna()
            
            # 计算夏普比率 (假设无风险利率为 2%)
            annual_returns = returns.mean() * 252  # 假设一年有252个交易日
            annual_volatility = returns.std() * (252**0.5)
            sharpe_ratio = (annual_returns - 0.02) / annual_volatility if annual_volatility != 0 else 0
            
            # 计算最大回撤
            rolling_max = fund_data['单位净值'].cummax()
            daily_drawdown = (fund_data['单位净值'] - rolling_max) / rolling_max
            max_drawdown = daily_drawdown.min() * -1
            
            self.fund_data = {
                'fund_code': self.fund_code,
                'latest_nav': fund_data['单位净值'].iloc[-1],
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            print(f"基金 {self.fund_code} 数据已获取：{self.fund_data}")
        except Exception as e:
            print(f"获取基金数据失败: {e}")
            return False
        return True

    def get_market_sentiment(self):
        """
        模拟获取市场情绪，一个简单的判断。
        例如：如果大盘指数过去一周下跌，则判断为悲观。
        """
        print("正在获取市场情绪数据...")
        try:
            # 获取上证指数数据，用于判断市场趋势
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            last_week_data = index_data.iloc[-7:]
            
            # 简单判断大盘涨跌
            if last_week_data['close'].iloc[-1] < last_week_data['close'].iloc[0]:
                sentiment = 'pessimistic'
                trend = 'bearish'
            else:
                sentiment = 'optimistic'
                trend = 'bullish'
            
            self.market_data = {'sentiment': sentiment, 'trend': trend}
            print(f"市场情绪数据已获取：{self.market_data}")
        except Exception as e:
            print(f"获取市场数据失败: {e}")
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown'}
            return False
        return True

    def make_decision(self, personal_strategy: dict) -> str:
        """
        根据所有已获取的数据和个人策略，做出投资决策。
        """
        # 确保数据已获取
        if not self.fund_data or not self.market_data:
            return "数据获取不完整，无法给出明确建议。"

        market_trend = self.market_data.get('trend', 'unknown')
        fund_drawdown = self.fund_data.get('max_drawdown', float('inf'))
        invest_horizon = personal_strategy.get('horizon', 'unknown')
        sharpe_ratio = self.fund_data.get('sharpe_ratio', 0)

        # 决策逻辑
        if invest_horizon == 'long-term':
            if market_trend == 'bearish':
                if fund_drawdown <= 0.2:
                    return f"市场处于熊市，但该基金回撤控制在 {fund_drawdown:.2f} 且您的投资期限较长，是长期布局的好时机。建议分批买入。"
                else:
                    return f"市场下跌，且该基金回撤高达 {fund_drawdown:.2f}，风险较高。建议保持观望或选择抗跌能力更强的基金。"
            else: # 牛市
                if sharpe_ratio > 1.5:
                    return f"市场处于牛市，该基金夏普比率高达 {sharpe_ratio:.2f}，表现优秀。适合继续持有或增加投资。"
                else:
                    return "市场上涨，但该基金表现平平。建议深入研究其投资策略，或寻找更具潜力的基金。"

        elif invest_horizon == 'short-term':
            if sharpe_ratio > 1.5 and market_trend == 'bullish':
                return f"市场处于牛市，该基金夏普比率高达 {sharpe_ratio:.2f}，表明其在承担风险时有优秀回报。适合短期持有。"
            else:
                return "当前市场或基金指标不适合短期投资，风险较高。建议保持谨慎。"

        return "个人投资策略与当前市场状况不匹配，请重新审视。"

# --- 脚本使用示例 ---
if __name__ == '__main__':
    # 你想分析的基金代码，例如：易方达蓝筹精选混合A
    FUND_CODE = "005827" 
    
    # 1. 实例化分析器
    analyzer = FundAnalyzer(fund_code=FUND_CODE)
    
    # 2. 获取真实数据
    analyzer.get_real_time_fund_data()
    analyzer.get_market_sentiment()
    
    # 3. 定义你的个人投资策略
    my_personal_strategy = {
        'horizon': 'long-term',  # 可以改为 'short-term'
        'risk_tolerance': 'medium'
    }

    # 4. 做出决策并打印结果
    decision = analyzer.make_decision(my_personal_strategy)
    print("\n--- 基金投资分析报告 ---")
    print(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"分析基金: {FUND_CODE}")
    print(f"当前市场趋势: {analyzer.market_data.get('trend')}")
    print(f"基金夏普比率: {analyzer.fund_data.get('sharpe_ratio', 0):.2f}")
    print(f"基金最大回撤: {analyzer.fund_data.get('max_drawdown', 0):.2f}")
    print("-" * 25)
    print("投资决策:")
    print(decision)
