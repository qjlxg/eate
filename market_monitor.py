import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import io

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
        """从 analysis_report.md 提取推荐基金代码"""
        logger.info("正在解析 %s 获取推荐基金代码...", self.report_file)
        if not os.path.exists(self.report_file):
            logger.error("报告文件 %s 不存在", self.report_file)
            raise FileNotFoundError(f"{self.report_file} 不存在")
        
        try:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取推荐基金表格
            pattern = r'\| *(\d{6}) *\|.*?\| *(\d+\.?\d*) *\|'
            matches = re.findall(pattern, content)
            self.fund_codes = [code for code, score in matches if float(score) >= 30][:20]
            logger.info("提取到 %d 个推荐基金 (限制前20): %s", len(self.fund_codes), self.fund_codes)
        except Exception as e:
            logger.error("解析 %s 失败: %s", self.report_file, str(e))
            raise

    def _get_fund_data(self, fund_code):
        """通过网络爬虫获取基金历史净值数据并计算技术指标"""
        logger.info("正在获取基金 %s 的净值数据...", fund_code)
        
        try:
            url = f"http://www.dayfund.cn/fundvalue/{fund_code}.html"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # 设置10秒超时
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'  # 确保正确解码中文

            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='mt1 clear')
            
            if not table:
                logger.warning("未找到基金 %s 的历史净值表格，可能网页结构已变更", fund_code)
                self.fund_data[fund_code] = None
                return

            try:
                df = pd.read_html(io.StringIO(str(table)), flavor='lxml')[0]
                df.columns = ['净值日期', '基金代码', '基金名称', '最新单位净值', '最新累计净值', '上期单位净值', '上期累计净值', '当日增长值', '当日增长率']
                df['净值日期'] = pd.to_datetime(df['净值日期'], format='%Y-%m-%d', errors='coerce')
                df['最新单位净值'] = pd.to_numeric(df['最新单位净值'], errors='coerce')
                df.set_index('净值日期', inplace=True)
                df.sort_index(inplace=True)
            except Exception as e:
                logger.error("解析基金 %s 的表格数据失败: %s", fund_code, str(e))
                self.fund_data[fund_code] = None
                return

            if len(df) < 50:
                logger.warning("基金 %s 历史净值数据不足50天，无法计算指标", fund_code)
                self.fund_data[fund_code] = None
                return

            # 计算RSI
            delta = df['最新单位净值'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # 计算MA50
            ma50 = df['最新单位净值'].rolling(window=50).mean()
            ma_ratio = df['最新单位净值'].iloc[-1] / ma50.iloc[-1]

            self.fund_data[fund_code] = {
                'latest_net_value': df['最新单位净值'].iloc[-1],
                'rsi': rsi.iloc[-1],
                'ma_ratio': ma_ratio
            }
            logger.info("成功获取并计算基金 %s 的技术指标", fund_code)

        except requests.exceptions.RequestException as e:
            logger.error("获取基金 %s 数据失败 (网络错误): %s", fund_code, str(e))
            self.fund_data[fund_code] = None
        except Exception as e:
            logger.error("获取基金 %s 数据失败 (其他错误): %s", fund_code, str(e))
            self.fund_data[fund_code] = None

    def run(self):
        """主执行函数"""
        try:
            # 提取基金代码
            self._parse_report()
            
            # 获取基金数据
            for fund_code in self.fund_codes:
                self._get_fund_data(fund_code)
            
            # 生成报告
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 市场情绪与技术指标监控报告\n\n")
                f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## 推荐基金技术指标\n")
                f.write("| 基金代码 | 最新净值 | RSI | 净值/MA50 | 投资建议 |\n")
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
                        f.write(f"| {fund_code} | {data['latest_net_value']:.4f} | {rsi:.2f} | {ma_ratio:.2f} | {advice} |\n")
                    else:
                        f.write(f"| {fund_code} | 数据获取失败 | - | - | 观察 |\n")
            
            logger.info("报告生成完成: %s", self.output_file)

        except FileNotFoundError as e:
            logger.error("文件缺失: %s", e)
            raise
        except Exception as e:
            logger.error("运行过程中发生错误: %s", e)
            raise

if __name__ == "__main__":
    monitor = MarketMonitor()
    monitor.run()
