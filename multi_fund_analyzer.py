import pandas as pd
import akshare as ak
from datetime import datetime
import numpy as np
import time

class FundAnalyzer:
    """
    一个用于综合分析基金投资的工具类。
    能从 CSV 文件导入基金代码，结合收益率、排名、实时数据和基金经理数据进行投资决策。
    """
    def __init__(self, cache_data: bool = True, risk_free_rate: float = 0.02):
        self.fund_data = {}
        self.fund_info = {}  # 存储 CSV 数据
        self.market_data = {}
        self.manager_data = {}
        self.analysis_report = []
        self.cache_data = cache_data
        self.risk_free_rate = risk_free_rate
        self.cache = {}

    def _log(self, message: str):
        """将日志信息添加到报告列表中"""
        print(message)
        self.analysis_report.append(message)

    def get_real_time_fund_data(self, fund_code: str):
        """获取单个基金的实时数据（净值、夏普比率、最大回撤）"""
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
                'latest_nav': float(fund_data['单位净值'].iloc[-1]),
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
        """获取市场情绪（仅调用一次，基于上证指数）"""
        if self.market_data:
            self._log("使用缓存的市场情绪数据")
            return True
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
            return True
        except Exception as e:
            self._log(f"获取市场数据失败: {e}")
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown'}
            return False

    def get_fund_manager_data(self, fund_code: str):
        """
        获取基金经理数据（使用更稳定的 ak.fund_manager_info_em 接口）
        """
        self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
        try:
            # 使用更稳定的接口，并通过基金代码查找经理
            manager_df = ak.fund_manager_info_em(fund=fund_code)
            
            if manager_df.empty:
                self._log(f"未找到基金 {fund_code} 的基金经理数据。")
                self.manager_data[fund_code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
                return False

            # 获取最新任职的基金经理
            latest_manager = manager_df.sort_values(by='上任日期', ascending=False).iloc[0]
            
            # 确保字段存在且类型正确
            name = latest_manager.get('姓名', 'N/A')
            tenure_days = latest_manager.get('任职天数', np.nan)
            cumulative_return = latest_manager.get('累计回报', '0%')
            cumulative_return = float(str(cumulative_return).replace('%', '')) if isinstance(cumulative_return, str) else float(cumulative_return)
            
            self.manager_data[fund_code] = {
                'name': name,
                'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                'cumulative_return': cumulative_return
            }
            self._log(f"基金 {fund_code} 经理数据已获取：{self.manager_data[fund_code]}")
            return True
        except Exception as e:
            self._log(f"获取基金 {fund_code} 经理数据失败: {e}")
            self.manager_data[fund_code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
            return False

    def make_decision(self, fund_code: str, personal_strategy: dict) -> str:
        """根据基金数据、CSV 数据和个人策略做出投资决策"""
        self._log(f"开始做出 {fund_code} 的投资决策:")
        if fund_code not in self.fund_data or not self.market_data:
            return "数据获取不完整，无法给出明确建议。"

        market_trend = self.market_data.get('trend', 'unknown')
        fund_drawdown = self.fund_data[fund_code].get('max_drawdown', float('inf'))
        invest_horizon = personal_strategy.get('horizon', 'unknown')
        risk_tolerance = personal_strategy.get('risk_tolerance', 'medium')
        sharpe_ratio = self.fund_data[fund_code].get('sharpe_ratio', 0)
        
        # 提取 CSV 数据
        fund_info = self.fund_info.get(fund_code, {})
        rose_3y = fund_info.get('rose(3y)', np.nan)
        rank_r_3y = fund_info.get('rank_r(3y)', np.nan)
        fund_name = fund_info.get('名称', '未知')

        # 基金经理数据
        manager_trust = False
        manager_return = self.manager_data.get(fund_code, {}).get('cumulative_return', np.nan)
        tenure_years = self.manager_data.get(fund_code, {}).get('tenure_years', np.nan)

        if pd.notna(manager_return):
            if tenure_years > 5 or manager_return > 20:
                manager_trust = True
            self._log(f"基金经理 {self.manager_data[fund_code]['name']} 任职 {tenure_years:.2f} 年，累计回报 {manager_return:.2f}%，增加信任度。")

        # 决策逻辑
        if invest_horizon == 'long-term':
            if market_trend == 'bearish':
                if fund_drawdown <= 0.2 and (risk_tolerance in ['medium', 'high'] or manager_trust):
                    return f"市场熊市，{fund_name} 回撤 {fund_drawdown:.2f} 控制良好，适合长期布局。建议分批买入。"
                else:
                    return f"市场熊市，{fund_name} 回撤 {fund_drawdown:.2f}，风险较高。建议观望或选择更稳健的基金（如债券型基金）。"
            else:  # bullish or neutral
                is_top_performer = (sharpe_ratio > 1.0 or pd.notna(rose_3y) and rose_3y > 50 or manager_trust)
                if is_top_performer and pd.notna(rank_r_3y) and rank_r_3y < 0.05 and risk_tolerance != 'low':
                    return (f"市场 {market_trend}，{fund_name} 表现优异，3年回报 {rose_3y:.2f}% "
                            f"（排名前 {rank_r_3y*100:.2f}%）。适合继续持有或加仓。")
                else:
                    return (f"市场 {market_trend}，但 {fund_name} 表现平平（夏普 {sharpe_ratio:.2f}，3年排名 {rank_r_3y*100:.2f}%）。建议评估其他排名更高的基金。")
        elif invest_horizon == 'short-term':
            if sharpe_ratio > 1.5 and market_trend == 'bullish' and risk_tolerance != 'low':
                return f"市场牛市，{fund_name} 夏普比率 {sharpe_ratio:.2f}，适合短期投资。建议适量买入。"
            else:
                return f"市场 {market_trend} 或 {fund_name} 不适合短期投资。建议保持谨慎。"
        return "投资策略与市场状况不匹配，请重新审视。"

    def analyze_multiple_funds(self, csv_url: str, personal_strategy: dict, code_column: str = '代码', max_funds: int = None):
        """
        批量分析 CSV 文件中的基金代码，结合 CSV 数据、实时数据和经理数据。
        """
        self._log("正在从 CSV 导入基金代码列表...")
        try:
            # 添加 encoding 参数修复编码问题
            funds_df = pd.read_csv(csv_url, encoding='gbk')
            self._log(f"导入成功，共 {len(funds_df)} 个基金代码")
            
            # 存储 CSV 数据（以代码为键）
            funds_df[code_column] = funds_df[code_column].astype(str).str.zfill(6)
            self.fund_info = funds_df.set_index(code_column).to_dict('index')

            fund_codes = funds_df[code_column].unique().tolist()
            if max_funds:
                fund_codes = fund_codes[:max_funds]
                self._log(f"限制分析前 {max_funds} 个基金：{fund_codes}...")
        except Exception as e:
            self._log(f"导入 CSV 失败: {e}")
            return None

        self.get_market_sentiment()

        results = []
        for code in fund_codes:
            self.get_real_time_fund_data(code)
            self.get_fund_manager_data(code)
            decision = self.make_decision(code, personal_strategy)
            
            fund_info = self.fund_info.get(code, {})
            results.append({
                'fund_code': code,
                'fund_name': fund_info.get('名称', '未知'),
                'rose_3y': fund_info.get('rose(3y)', np.nan),
                'rank_r_3y': fund_info.get('rank_r(3y)', np.nan),
                'latest_nav': self.fund_data.get(code, {}).get('latest_nav', np.nan),
                'sharpe_ratio': self.fund_data.get(code, {}).get('sharpe_ratio', np.nan),
                'max_drawdown': self.fund_data.get(code, {}).get('max_drawdown', np.nan),
                'manager_name': self.manager_data.get(code, {}).get('name', 'N/A'),
                'manager_return': self.manager_data.get(code, {}).get('cumulative_return', np.nan),
                'tenure_years': self.manager_data.get(code, {}).get('tenure_years', np.nan),
                'market_trend': self.market_data.get('trend', 'unknown'),
                'decision': decision
            })
            time.sleep(1) # 增加延迟，避免请求过快被封

        results_df = pd.DataFrame(results)
        
        print("\n--- 批量基金分析报告 ---")
        print(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"市场趋势: {self.market_data.get('trend', 'unknown')}")
        
        print("\n所有基金分析结果:")
        print(results_df[['fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'sharpe_ratio', 'max_drawdown', 'manager_name', 'decision']].to_string(index=False))

        top_funds = results_df[results_df['rank_r_3y'] < 0.01].sort_values('rank_r_3y')
        if not top_funds.empty:
            print("\n--- 推荐基金（3年排名前 1%）---")
            print(top_funds[['fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'sharpe_ratio', 'max_drawdown', 'manager_name', 'manager_return', 'tenure_years', 'decision']].to_string(index=False))
        else:
            print("\n没有基金满足 3 年排名前 1% 的条件。")
        print("-" * 25)
        return results_df

if __name__ == '__main__':
    CSV_URL = "https://raw.githubusercontent.com/qjlxg/rep/main/recommended_cn_funds.csv"
    analyzer = FundAnalyzer(risk_free_rate=0.03)
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }
    results_df = analyzer.analyze_multiple_funds(CSV_URL, my_personal_strategy, code_column='代码', max_funds=10)
