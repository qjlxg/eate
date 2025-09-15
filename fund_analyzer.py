import pandas as pd

class FundAnalyzer:
    """
    一个用于综合分析基金投资的工具类。
    """
    def __init__(self):
        self.market_data = {}
        self.fund_data = {}
        self.manager_data = {}

    def get_market_data(self, trend: str, sentiment: str):
        """
        模拟获取市场环境数据。
        
        参数:
        trend (str): 市场趋势，例如 'bullish' (牛市) 或 'bearish' (熊市)。
        sentiment (str): 市场情绪，例如 'optimistic' (乐观) 或 'pessimistic' (悲观)。
        """
        self.market_data = {
            'trend': trend,
            'sentiment': sentiment
        }
        print(f"市场环境数据已获取：{self.market_data}")

    def get_fund_data(self, fund_code: str, sharpe_ratio: float, max_drawdown: float):
        """
        获取基金的关键风险收益数据。
        
        参数:
        fund_code (str): 基金代码。
        sharpe_ratio (float): 夏普比率。
        max_drawdown (float): 最大回撤。
        """
        self.fund_data = {
            'fund_code': fund_code,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        print(f"基金 {fund_code} 的关键指标已获取：{self.fund_data}")

    def get_manager_data(self, name: str, style: str, performance_df: pd.DataFrame):
        """
        获取基金经理的数据，包括历史业绩。
        
        参数:
        name (str): 基金经理姓名。
        style (str): 投资风格。
        performance_df (pd.DataFrame): 包含历史业绩的 Pandas DataFrame。
        """
        self.manager_data = {
            'name': name,
            'style': style,
            'performance': performance_df
        }
        print(f"基金经理 {name} 的数据已获取。")

    def make_decision(self, personal_strategy: dict) -> str:
        """
        根据所有已获取的数据和个人策略，做出投资决策。
        
        参数:
        personal_strategy (dict): 包含个人投资策略的字典，例如 {'horizon': 'long-term', 'risk_tolerance': 'medium'}。
        
        返回 (str): 投资建议。
        """
        market_trend = self.market_data.get('trend', 'unknown')
        fund_drawdown = self.fund_data.get('max_drawdown', float('inf'))
        invest_horizon = personal_strategy.get('horizon', 'unknown')

        # 决策逻辑
        if market_trend == 'bearish' and invest_horizon == 'long-term':
            if fund_drawdown <= 0.2:
                return "市场处于下跌趋势，但您的投资期限较长且该基金回撤控制较好，是长期布局的好时机。建议分批买入。"
            else:
                return "市场下跌，且该基金回撤较大，风险较高。建议保持观望或选择抗跌能力更强的基金。"
        
        elif market_trend == 'bullish' and invest_horizon == 'short-term':
            if self.fund_data.get('sharpe_ratio', 0) > 1.5:
                return "市场上涨，且该基金夏普比率较高，表明其在承担风险时有优秀的回报。适合短期持有。"
            else:
                return "市场上涨，但该基金表现平平。建议选择更具成长潜力的基金。"

        return "数据不足或策略不匹配，无法给出明确建议。请完善您的信息。"

# --- 脚本使用示例 ---
if __name__ == '__main__':
    # 1. 实例化分析器
    analyzer = FundAnalyzer()

    # 2. 准备数据
    # 模拟基金经理的历史业绩数据
    manager_performance = pd.DataFrame({
        'year': [2022, 2023, 2024],
        'return_rate': [0.1, -0.05, 0.25]
    })

    # 3. 传入数据到分析器
    analyzer.get_market_data(trend='bearish', sentiment='pessimistic')
    analyzer.get_fund_data(fund_code='001234', sharpe_ratio=1.2, max_drawdown=0.18)
    analyzer.get_manager_data(name='王雷', style='value', performance_df=manager_performance)

    # 4. 定义个人投资策略
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }

    # 5. 做出决策并打印结果
    decision = analyzer.make_decision(my_personal_strategy)
    print("\n--- 投资决策 ---")
    print(decision)
