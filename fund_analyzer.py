import pandas as pd
import akshare as ak
from datetime import datetime
import time

class MultiFundAnalyzer:
    """
    一个用于综合分析多个基金的工具类。
    能实时获取多支基金的关键数据，并进行横向对比。
    """
    def __init__(self, fund_list: list, risk_free_rate: float = 0.03):
        self.fund_list = fund_list
        self.risk_free_rate = risk_free_rate
        self.analysis_results = []
        self.failed_funds = []
        self.market_data = {}

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

    def analyze_single_fund(self, fund_code: str):
        """
        分析单个基金，获取其所有关键数据。
        """
        self._log(f"\n--- 开始分析基金: {fund_code} ---")
        try:
            # 获取基金净值数据
            fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            fund_data.set_index('净值日期', inplace=True)
            returns = fund_data['单位净值'].pct_change().dropna()
            
            # 计算关键指标
            annual_returns = returns.mean() * 252
            annual_volatility = returns.std() * (252**0.5)
            sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
            
            rolling_max = fund_data['单位净值'].cummax()
            daily_drawdown = (fund_data['单位净值'] - rolling_max) / rolling_max
            max_drawdown = daily_drawdown.min() * -1
            
            # 获取基金经理信息
            manager_data = {}
            try:
                manager_info = ak.fund_manager(symbol=fund_code)
                if not manager_info.empty:
                    manager_info = manager_info.sort_values(by='上任日期', ascending=False).iloc[0]
                    manager_data = {
                        'name': manager_info['基金经理'],
                        'tenure_years': manager_info['任职天数'] / 365.0,
                        'cumulative_return': float(str(manager_info['累计回报']).replace('%', ''))
                    }
            except Exception as e:
                self._log(f"获取基金经理数据失败 for {fund_code}: {e}")

            result = {
                '基金代码': fund_code,
                '夏普比率': sharpe_ratio,
                '最大回撤': max_drawdown,
                '基金经理': manager_data.get('name', 'N/A'),
                '任职年限': manager_data.get('tenure_years', 'N/A'),
                '任职总回报(%)': manager_data.get('cumulative_return', 'N/A')
            }
            self.analysis_results.append(result)
            self._log(f"基金 {fund_code} 数据分析完成。")
            
        except Exception as e:
            self.failed_funds.append(fund_code)
            self._log(f"分析基金 {fund_code} 失败: {e}")

    def run_analysis(self):
        """
        运行多基金分析的主流程。
        """
        self._log(f"\n--- 开始多基金分析，共 {len(self.fund_list)} 支 ---")
        self.get_market_sentiment()
        
        for fund_code in self.fund_list:
            self.analyze_single_fund(fund_code)
            time.sleep(1) # 增加延迟，避免请求过快被封

        self._log("\n--- 所有基金分析已完成 ---")
        self._log("--- 生成横向对比报告 ---")
        
        if not self.analysis_results:
            self._log("没有成功分析的基金。")
            return
            
        results_df = pd.DataFrame(self.analysis_results)
        
        # 格式化输出，便于阅读
        results_df['夏普比率'] = results_df['夏普比率'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        results_df['最大回撤'] = results_df['最大回撤'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        results_df['任职年限'] = results_df['任职年限'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        
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

    # 创建分析器实例
    analyzer = MultiFundAnalyzer(fund_list=FUND_LIST)
    
    # 运行分析流程
    analyzer.run_analysis()
