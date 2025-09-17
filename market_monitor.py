import pandas as pd
import numpy as np
import akshare as ak
import re
import os
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    filename='market_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketMonitor:
    def __init__(self, report_file='analysis_report.md', output_file='market_monitor_report.md'):
        self.report_file = report_file
        self.output_file = output_file
        self.fund_codes = []
        self.fund_data = {}
        self.market_data = {}
        self.technical_indicators = {}

    def _parse_report(self):
        """从analysis_report.md提取推荐基金代码"""
        logger.info("正在解析 %s 获取推荐基金代码...", self.report_file)
        if not os.path.exists(self.report_file):
            logger.error("报告文件 %s 不存在", self.report_file)
            raise FileNotFoundError(f"{self.report_file} 不存在")
        
        with open(self.report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取推荐基金表格
        # 匹配 "| 基金代码 | ... | 分数 |"
        pattern = r'\| *(\d{6}) *\|.*?\| *(\d+\.?\d*) *\|'
        matches = re.findall(pattern, content)
        
        # 提取符合条件的基金代码并进行去重
        # Modification: Added set() to deduplicate fund codes after extraction.
        # 修正：增加了 set()，用于在提取后对基金代码进行去重。
        unique_fund_codes = set()
        for code, score in matches:
            if float(score) >= 4.0:
                unique_fund_codes.add(code)

        self.fund_codes = sorted(list(unique_fund_codes))
        
        logger.info("提取到 %d 个推荐基金: %s", len(self.fund_codes), self.fund_codes)
        
    def _get_market_data(self):
        """获取市场情绪数据"""
        # 假设从某个API获取市场情绪数据
        # 实际应用中需要替换为真实的数据源
        logger.info("正在获取市场情绪数据...")
        self.market_data = {
            'sentiment': 'bullish',
            'score': 5
        }
        logger.info("市场情绪数据获取成功: %s", self.market_data)

    def _get_fund_data(self, fund_code):
        """获取基金净值数据并计算技术指标"""
        logger.info("获取基金 %s 的净值数据...", fund_code)
        try:
            # 修正: akshare库中参数名应为symbol，而不是fund
            # Fix: The parameter name in akshare should be 'symbol', not 'fund'
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")
            df['净值日期'] = pd.to_datetime(df['净值日期'])
            df.set_index('净值日期', inplace=True)
            
            # 计算RSI
            close = df['累计净值'].astype(float)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 计算MA50
            ma50 = close.rolling(window=50).mean()
            
            # 计算净值与MA50的比率
            ma_ratio = close.iloc[-1] / ma50.iloc[-1]
            
            self.fund_data[fund_code] = {
                'latest_nav': close.iloc[-1],
                'rsi': rsi.iloc[-1],
                'ma_ratio': ma_ratio
            }
            logger.info("基金 %s 数据获取并计算成功", fund_code)
        except Exception as e:
            logger.error("获取基金 %s 数据失败: %s", fund_code, e)
            self.fund_data[fund_code] = None

    def run(self):
        """主执行流程"""
        try:
            self._parse_report()
            self._get_market_data()
            
            for fund_code in self.fund_codes:
                self._get_fund_data(fund_code)
            
            # 生成报告
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 市场情绪与技术指标监控报告\n\n")
                f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## 市场情绪\n- 情绪: {self.market_data['sentiment']}\n- 分数: {self.market_data['score']}\n\n")
                
                f.write("## 推荐基金技术指标\n")
                f.write("| 基金代码 | 最新净值 | RSI | 净值/MA50 | 投资建议 | \n")
                f.write("|----------|----------|-----|-----------|----------|\n")
                
                for fund_code in self.fund_codes:
                    if fund_code in self.fund_data and self.fund_data[fund_code]:
                        data = self.fund_data[fund_code]
                        rsi = data['rsi']
                        ma_ratio = data['ma_ratio']
                        advice = (
                            "等待回调" if rsi > 70 or ma_ratio > 1.2 else
                            "可分批买入" if 30 <= rsi <= 70 and 0.8 <= ma_ratio <= 1.2 else
                            "可加仓" if rsi < 30 else "观察"
                        )
                        f.write(f"| {fund_code} | {data['latest_nav']:.4f} | {rsi:.2f} | {ma_ratio:.2f} | {advice} |\n")
                    else:
                        f.write(f"| {fund_code} | - | - | - | 无法获取数据 |\n")
            
            logger.info("报告生成成功: %s", self.output_file)
            print(f"报告生成成功: {self.output_file}")

        except FileNotFoundError as e:
            print(e)
            
        except Exception as e:
            logger.error("脚本运行失败: %s", e)
            print(f"脚本运行失败: {e}")

if __name__ == "__main__":
    monitor = MarketMonitor()
    monitor.run()
