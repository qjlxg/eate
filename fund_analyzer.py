import pandas as pd
import akshare as ak
from datetime import datetime
import time

class MultiFundAnalyzer:
    """
    一个用于综合分析多个基金的工具类。
    能实时获取多支基金的关键数据，并进行横向对比，并提供投资决策。
    """
    def __init__(self, fund_list: list, risk_free_rate: float = 0.03):
        self.fund_list = fund_list
        self.risk_free_rate = risk_free_rate
        self.analysis_results = []
        self.failed_funds = []
        self.market_data = {}
        # 初始化基金经理数据缓存，避免重复请求
        self._manager_cache = {}

    def _log(self, message: str):
        """将日志信息打印到控制台"""
        print(message)

    def get_market_sentiment(self):
        """
        获取市场情绪，一个简单的判断。
        """
        self._log("正在获取市场情绪数据...")
        try:
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            last_week_data = index_data.iloc[-7:]
            
            price_change = last_week_data['close'].iloc[-1] / last_week_data['close'].iloc[0] - 1
            volume_change = last_week_data['volume'].mean() / last_week_data['volume'].iloc[:-1].mean() - 1
            
            if price_change > 0.01 and volume_change > 0:
                sentiment, trend = 'optimistic', 'bullish'
            elif price_change < -0.01:
                sentiment, trend = 'pessimistic', 'bearish'
            else:
                sentiment, trend = 'neutral', 'neutral'
            
            self.market_data = {'sentiment': sentiment, 'trend': trend}
            self._log(f"市场情绪数据已获取：{self.market_data}")
        except Exception as e:
            self._log(f"获取市场数据失败: {e}")
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown'}

    def analyze_single_fund(self, fund_code: str, strategy: dict):
        """
        分析单个基金，获取其所有关键数据，并做出投资决策。
        """
        self._log(f"\n--- 开始分析基金: {fund_code} ---")
        try:
            # 获取基金净值数据
            fund_data_raw = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            fund_data_raw['净值日期'] = pd.to_datetime(fund_data_raw['净值日期'])
            fund_data_raw.set_index('净值日期', inplace=True)
            returns = fund_data_raw['单位净值'].pct_change().dropna()
            
            # 计算关键指标
            annual_returns = returns.mean() * 252
            annual_volatility = returns.std() * (252**0.5)
            sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
            
            rolling_max = fund_data_raw['单位净值'].cummax()
            daily_drawdown = (fund_data_raw['单位净值'] - rolling_max) / rolling_max
            max_drawdown = daily_drawdown.min() * -1
            
            # 获取基金经理信息
            manager_data = self._get_fund_manager_data(fund_code)
            
            # 做出投资决策
            decision = self._make_decision(
                fund_data={'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown},
                manager_data=manager_data,
                strategy=strategy
            )

            result = {
                '基金代码': fund_code,
                '夏普比率': sharpe_ratio,
                '最大回撤': max_drawdown,
                '基金经理': manager_data.get('name', 'N/A'),
                '任职年限': manager_data.get('tenure_years', 'N/A'),
                '任职总回报(%)': manager_data.get('cumulative_return', 'N/A'),
                '投资决策': decision
            }
            self.analysis_results.append(result)
            self._log(f"基金 {fund_code} 数据分析完成。")
            
        except Exception as e:
            self.failed_funds.append(fund_code)
            self._log(f"分析基金 {fund_code} 失败: {e}")

    def _get_fund_manager_data(self, fund_code: str):
        """
        获取单个基金的经理数据，并使用缓存。
        """
        if fund_code in self._manager_cache:
            return self._manager_cache[fund_code]

        try:
            # 修复后的基金经理数据获取方式
            manager_info = ak.fund_manager_info_em(fund=fund_code)
            
            if manager_info.empty:
                self._log("未找到基金经理数据。")
                return {}

            manager_info = manager_info.sort_values(by='上任日期', ascending=False).iloc[0]
            cumulative_return = float(str(manager_info['累计回报']).replace('%', ''))
            
            manager_data = {
                'name': manager_info['基金经理'],
                'tenure_years': manager_info['任职天数'] / 365.0,
                'cumulative_return': cumulative_return
            }
            self._manager_cache[fund_code] = manager_data
            return manager_data
        except Exception as e:
            self._log(f"获取基金经理数据失败 for {fund_code}: {e}")
            return {}

    def _make_decision(self, fund_data: dict, manager_data: dict, strategy: dict) -> str:
        """
        根据基金数据、市场情绪和个人策略做出决策。
        """
        if not self.market_data:
            return "数据获取不完整，无法给出明确建议。"
        
        market_trend = self.market_data.get('trend', 'unknown')
        fund_drawdown = fund_data.get('max_drawdown', float('inf'))
        invest_horizon = strategy.get('horizon', 'unknown')
        risk_tolerance = strategy.get('risk_tolerance', 'medium')
        sharpe_ratio = fund_data.get('sharpe_ratio', 0)
        
        manager_trust = False
        if manager_data:
            tenure_years = manager_data.get('tenure_years', 0)
            manager_return = manager_data.get('cumulative_return', -1)
            if tenure_years > 5 or manager_return > 20:
                manager_trust = True
        
        # 决策逻辑...
        if invest_horizon == 'long-term':
            if market_trend == 'bearish':
                if fund_drawdown <= 0.2 and (risk_tolerance in ['medium', 'high'] or manager_trust):
                    return f"市场处于熊市，但基金回撤控制在 {fund_drawdown:.2f}，且经理能力较强。建议分批买入。"
                else:
                    return f"市场下跌，基金回撤 {fund_drawdown:.2f}，风险较高。建议观望或选择更稳健的基金（如债券型基金）。"
            elif market_trend == 'bullish':
                if (sharpe_ratio > 1.5 or manager_trust) and risk_tolerance != 'low':
                    return f"市场处于牛市，基金夏普比率 {sharpe_ratio:.2f}，经理累计回报 {manager_return:.2f}%。适合继续持有或加仓。"
                else:
                    return f"市场上涨，但基金表现一般（夏普比率 {sharpe_ratio:.2f}）或风险偏好较低。建议评估其他基金（如指数基金）。"
            else: # neutral
                if sharpe_ratio > 1.0 and manager_trust and risk_tolerance != 'low':
                    return f"市场中性，基金夏普比率 {sharpe_ratio:.2f}，经理能力较强。适合适量投资。"
                else:
                    return f"市场中性，基金表现一般（夏普比率 {sharpe_ratio:.2f}，回撤 {fund_drawdown:.2f}）。建议评估其他低回撤基金。"
        
        elif invest_horizon == 'short-term':
            if sharpe_ratio > 1.5 and market_trend == 'bullish' and risk_tolerance != 'low':
                return f"市场处于牛市，基金夏普比率 {sharpe_ratio:.2f}，适合短期投资（风险承受能力 {risk_tolerance}）。建议适量买入。"
            else:
                return f"市场或基金指标不适合短期投资（风险承受能力 {risk_tolerance}）。建议保持谨慎或选择短期债券基金。"
        
        return "投资策略与市场状况不匹配，请重新审视。"

    def run_analysis(self, personal_strategy: dict):
        """
        运行多基金分析的主流程。
        """
        self._log(f"\n--- 开始多基金分析，共 {len(self.fund_list)} 支 ---")
        self.get_market_sentiment()
        
        for fund_code in self.fund_list:
            self.analyze_single_fund(fund_code, personal_strategy)
            time.sleep(1) # 增加延迟，避免请求过快被封

        self._log("\n--- 所有基金分析已完成 ---")
        self._log("--- 生成横向对比报告 ---")
        
        if not self.analysis_results:
            self._log("没有成功分析的基金。")
            return
            
        results_df = pd.DataFrame(self.analysis_results)
        
        # 格式化输出，便于阅读
        for col in ['夏普比率', '最大回撤', '任职年限', '任职总回报(%)']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x)
        
        print("\n" + results_df.to_string(index=False))

        if self.failed_funds:
            self._log("\n--- 以下基金分析失败，请检查代码或网络 ---")
            self._log(f"失败列表: {', '.join(self.failed_funds)}")


# --- 脚本使用示例 ---
if __name__ == '__main__':
    try:
        # 读取 CSV 文件中的基金代码
        df = pd.read_csv('recommended_cn_funds.csv')
        FUND_LIST = df['代码'].astype(str).tolist()
    except FileNotFoundError:
        print("未找到 recommended_cn_funds.csv 文件，使用默认列表。")
        FUND_LIST = ["005827", "001938", "110011", "000001", "001639"]

    # 定义你的个人投资策略
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }

    # 创建分析器实例
    analyzer = MultiFundAnalyzer(fund_list=FUND_LIST)
    
    # 运行分析流程
    analyzer.run_analysis(my_personal_strategy)
