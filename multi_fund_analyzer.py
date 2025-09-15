import pandas as pd
import akshare as ak
from datetime import datetime
import numpy as np  # 用于处理 np.float64

class FundAnalyzer:
    """
    一个用于综合分析基金投资的工具类。
    能实时获取基金数据、基金经理数据并做出投资决策。
    支持批量分析 CSV 中的基金代码。
    """
    def __init__(self, cache_data: bool = True, risk_free_rate: float = 0.02):
        self.fund_data = {}
        self.market_data = {}
        self.manager_data = {}
        self.analysis_report = []
        self.cache_data = cache_data
        self.risk_free_rate = risk_free_rate
        self.cache = {}  # 缓存多个基金的数据

    def _log(self, message: str):
        """将日志信息添加到报告列表中"""
        print(message)
        self.analysis_report.append(message)

    def get_real_time_fund_data(self, fund_code: str):
        """
        使用 akshare 库获取单个基金的实时数据。
        """
        if self.cache_data and fund_code in self.cache.get('fund', {}):
            self._log(f"使用缓存数据 for 基金 {fund_code}")
            self.fund_data[fund_code] = self.cache['fund'][fund_code]
            return True

        self._log(f"正在获取基金 {fund_code} 的实时数据...")
        try:
            fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            fund_data.set_index('净值日期', inplace=True)
            
            returns = fund_data['单位净值'].pct_change().dropna()
            
            annual_returns = returns.mean() * 252
            annual_volatility = returns.std() * (252**0.5)
            sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
            
            rolling_max = fund_data['单位净值'].cummax()
            daily_drawdown = (fund_data['单位净值'] - rolling_max) / rolling_max
            max_drawdown = daily_drawdown.min() * -1
            
            self.fund_data[fund_code] = {
                'latest_nav': float(fund_data['单位净值'].iloc[-1]),  # 转换为 float
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown)
            }
            if self.cache_data:
                self.cache.setdefault('fund', {})[fund_code] = self.fund_data[fund_code]
            self._log(f"基金 {fund_code} 数据已获取：{self.fund_data[fund_code]}")
            return True
        except Exception as e:
            self._log(f"获取基金 {fund_code} 数据失败: {e}")
            self.fund_data[fund_code] = {'latest_nav': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
            return False

    def get_market_sentiment(self):
        """
        获取市场情绪，一个简单的判断（仅调用一次）。
        """
        if self.market_data:
            return True
        self._log("正在获取市场情绪数据...")
        try:
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            last_week_data = index_data.iloc[-7:]
            
            price_change = last_week_data['close'].iloc[-1] / last_week_data['close'].iloc[0] - 1
            if price_change > 0:
                sentiment = 'optimistic'
                trend = 'bullish'
            elif price_change < 0:
                sentiment = 'pessimistic'
                trend = 'bearish'
            else:
                sentiment = 'neutral'
                trend = 'neutral'
            
            self.market_data = {'sentiment': sentiment, 'trend': trend}
            self._log(f"市场情绪数据已获取：{self.market_data}")
            return True
        except Exception as e:
            self._log(f"获取市场数据失败: {e}")
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown'}
            return False

    def get_fund_manager_data(self, fund_code: str):
        """
        通过基金代码获取基金经理信息（修复接口）。
        使用 ak.fund_em_fund_manager 作为备用接口。
        """
        self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
        try:
            # 尝试使用 ak.fund_em_fund_manager（常见接口）
            manager_info = ak.fund_em_fund_manager(symbol=fund_code)
            if manager_info.empty:
                self._log(f"未找到基金 {fund_code} 的基金经理数据。")
                self.manager_data[fund_code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
                return False

            # 假设返回 DataFrame，选择最新经理
            manager_info = manager_info.sort_values(by='上任日期', ascending=False).iloc[0] if '上任日期' in manager_info.columns else manager_info.iloc[0]
            
            # 处理字段（根据实际返回调整）
            name = manager_info.get('姓名', manager_info.get('基金经理', 'N/A'))
            on_duty_days = manager_info.get('任职天数', 0)
            cumulative_return_str = manager_info.get('任职总回报', '0%')
            cumulative_return = float(cumulative_return_str.replace('%', '')) if isinstance(cumulative_return_str, str) else float(cumulative_return_str)
            
            self.manager_data[fund_code] = {
                'name': name,
                'tenure_years': on_duty_days / 365.0,
                'cumulative_return': cumulative_return
            }
            self._log(f"基金 {fund_code} 经理数据已获取：{self.manager_data[fund_code]}")
            return True
        except Exception as e:
            self._log(f"获取基金 {fund_code} 经理数据失败: {e}")
            self.manager_data[fund_code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
            return False
    
    def make_decision(self, fund_code: str, personal_strategy: dict) -> str:
        """
        根据单个基金的数据和个人策略，做出投资决策。
        """
        self._log(f"开始做出 {fund_code} 的投资决策:")

        if fund_code not in self.fund_data or not self.market_data:
            return "数据获取不完整，无法给出明确建议。"

        market_trend = self.market_data.get('trend', 'unknown')
        fund_drawdown = self.fund_data[fund_code].get('max_drawdown', float('inf'))
        invest_horizon = personal_strategy.get('horizon', 'unknown')
        sharpe_ratio = self.fund_data[fund_code].get('sharpe_ratio', 0)
        
        # 基金经理数据考量
        if fund_code in self.manager_data:
            manager_return = self.manager_data[fund_code].get('cumulative_return', -1)
            tenure_years = self.manager_data[fund_code].get('tenure_years', 0)
            if manager_return > 0:
                self._log(f"基金经理任职回报为 {manager_return:.2f}%，增加信任度。")
        else:
            manager_return = -1
            tenure_years = 0

        if invest_horizon == 'long-term':
            if market_trend == 'bearish':
                if fund_drawdown <= 0.2:
                    return f"市场熊市，但回撤 {fund_drawdown:.2f} 控制良好，是长期布局好时机。建议分批买入。"
                else:
                    return f"市场熊市，回撤高达 {fund_drawdown:.2f}，风险较高。建议观望。"
            else:
                if sharpe_ratio > 1.5 and manager_return > 0:
                    return f"市场牛市/中性，夏普比率 {sharpe_ratio:.2f}，经理回报 {manager_return:.2f}%。适合继续持有或增加投资。"
                else:
                    return f"市场牛市/中性，但基金表现平平（夏普 {sharpe_ratio:.2f}）。建议评估其他基金。"

        elif invest_horizon == 'short-term':
            if sharpe_ratio > 1.5 and market_trend == 'bullish':
                return f"市场牛市，夏普比率 {sharpe_ratio:.2f}，适合短期持有。"
            else:
                return "当前市场或基金不适合短期投资，风险较高。建议保持谨慎。"

        return "投资策略与市场不匹配，请重新审视。"

    def analyze_multiple_funds(self, csv_url: str, personal_strategy: dict, code_column: str = 'code'):
        """
        批量分析 CSV 文件中的基金代码。
        :param csv_url: CSV 文件 URL
        :param personal_strategy: 个人策略字典
        :param code_column: CSV 中的基金代码列名
        """
        self._log("正在从 CSV 导入基金代码列表...")
        try:
            # 直接从 URL 读取 CSV
            funds_df = pd.read_csv(csv_url)
            fund_codes = funds_df[code_column].unique().tolist()  # 提取唯一基金代码
            self._log(f"导入成功，共 {len(fund_codes)} 个基金代码：{fund_codes[:5]}...")  # 显示前5个
        except Exception as e:
            self._log(f"导入 CSV 失败: {e}")
            return None

        # 获取市场情绪（共享）
        self.get_market_sentiment()

        # 分析每个基金
        results = []
        for code in fund_codes:
            self.get_real_time_fund_data(code)
            self.get_fund_manager_data(code)
            decision = self.make_decision(code, personal_strategy)
            results.append({
                'fund_code': code,
                'latest_nav': self.fund_data.get(code, {}).get('latest_nav', np.nan),
                'sharpe_ratio': self.fund_data.get(code, {}).get('sharpe_ratio', np.nan),
                'max_drawdown': self.fund_data.get(code, {}).get('max_drawdown', np.nan),
                'manager_name': self.manager_data.get(code, {}).get('name', 'N/A'),
                'manager_return': self.manager_data.get(code, {}).get('cumulative_return', np.nan),
                'market_trend': self.market_data.get('trend', 'unknown'),
                'decision': decision
            })
            self._log("-" * 25)

        # 生成汇总报告表格
        results_df = pd.DataFrame(results)
        print("\n--- 批量基金分析报告 ---")
        print(results_df.to_string(index=False))
        print(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"市场趋势: {self.market_data.get('trend', 'unknown')}")
        print("-" * 25)
        return results_df

if __name__ == '__main__':
    CSV_URL = "https://raw.githubusercontent.com/qjlxg/rep/main/recommended_cn_funds.csv"
    
    analyzer = FundAnalyzer(risk_free_rate=0.03)
    
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }

    results_df = analyzer.analyze_multiple_funds(CSV_URL, my_personal_strategy, code_column='code')  # 假设列名为 'code'，如不同请调整
