import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
# 新增: 导入Selenium相关库
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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
        # 新增: 初始化Selenium驱动
        self.driver = None

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
            pattern = r'\| *(\d{6}) *\|.*?\| *(\d+\\.?\\d*) *\|'
            matches = re.findall(pattern, content)
            self.fund_codes = [code for code, _ in matches]
            logger.info("提取到 %d 个推荐基金: %s", len(self.fund_codes), self.fund_codes)
            
        except Exception as e:
            logger.error("解析报告文件失败: %s", e)
            raise

    def _get_fund_data_from_jijin(self, fund_code):
        """
        使用 Selenium 从基金速查网抓取基金历史净值数据
        """
        logger.info("正在获取基金 %s 的净值数据...", fund_code)
        
        # 使用 Selenium 抓取，代替原有的 requests + BeautifulSoup
        try:
            # 配置 Chrome 选项，支持无头模式（在无GUI环境下运行）
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            self.driver = webdriver.Chrome(options=options)
            
            # 使用基金速查网的URL
            url = f"http://www.jijinsucha.com/fundvalue/{fund_code}.html"
            self.driver.get(url)

            # 显式等待，确保净值表格加载完成，防止因网络延迟导致抓取失败
            wait = WebDriverWait(self.driver, 10)
            table_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.mt1')))
            
            # 使用 pandas 直接读取网页源代码中的表格
            df_list = pd.read_html(self.driver.page_source, flavor='html5lib')
            
            # 找到正确的表格，它通常是页面上的第一个表格或结构最完整的表格
            df = df_list[0]
            
            # 根据基金速查网的表格结构重新命名列
            df.columns = ['date', 'fund_code', 'fund_name', 'net_value', 'accumulated_net_value', 'prev_net_value', 'prev_accumulated_net_value', 'daily_growth_value', 'daily_growth_rate']
            
            df['date'] = pd.to_datetime(df['date'])
            df['net_value'] = pd.to_numeric(df['net_value'])
            
            # 只需要日期和净值列
            return df[['date', 'net_value']]

        except Exception as e:
            logger.warning("未能找到基金 %s 的历史净值表格，可能网页结构已变更或基金代码无效", fund_code)
            logger.error(f"Selenium 抓取失败: {e}")
            return None
        finally:
            if self.driver:
                self.driver.quit() # 确保在任何情况下都关闭浏览器实例

    def get_fund_data(self):
        """获取所有基金的数据"""
        for fund_code in self.fund_codes:
            # 调用新修改的抓取方法
            df = self._get_fund_data_from_jijin(fund_code)
            
            if df is not None and not df.empty:
                df = df.sort_values(by='date', ascending=True)
                # 计算 RSI
                delta = df['net_value'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()

                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

                # 计算 MA50
                ma50 = df['net_value'].rolling(window=50).mean()
                
                # 获取最新数据
                latest_data = df.iloc[-1]
                latest_net_value = latest_data['net_value']
                latest_rsi = rsi.iloc[-1]
                latest_ma50_ratio = latest_net_value / ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) and ma50.iloc[-1] != 0 else float('nan')

                self.fund_data[fund_code] = {
                    'latest_net_value': latest_net_value,
                    'rsi': latest_rsi,
                    'ma_ratio': latest_ma50_ratio
                }
                logger.info("成功获取并计算基金 %s 的技术指标: 净值=%.4f, RSI=%.2f, MA50比率=%.2f", 
                            fund_code, latest_net_value, latest_rsi, latest_ma50_ratio)
            else:
                self.fund_data[fund_code] = None
                logger.warning("基金 %s 数据获取失败，跳过计算", fund_code)

    def generate_report(self):
        """生成市场情绪与技术指标监控报告"""
        logger.info("正在生成市场监控报告...")
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

if __name__ == "__main__":
    try:
        monitor = MarketMonitor()
        monitor._parse_report()
        monitor.get_fund_data()
        monitor.generate_report()
    except Exception as e:
        logger.error("脚本运行失败: %s", e)
        # 确保在失败时也创建日志文件，以便GitHub Action捕捉到
        with open('market_monitor.log', 'a', encoding='utf-8') as f:
            f.write(f"\n[CRITICAL] 脚本因致命错误终止: {e}\n")