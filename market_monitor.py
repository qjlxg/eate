import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime, timedelta
import time
import random
import requests
import tenacity
from io import StringIO
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义本地数据存储目录
DATA_DIR = 'fund_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class MarketMonitor:
    def __init__(self, report_file='analysis_report.md', output_file='market_monitor_report.md'):
        self.report_file = report_file
        self.output_file = output_file
        self.fund_codes = []
        self.fund_data = {}

    def _parse_report(self):
        """从 analysis_report.md 提取推荐基金代码"""
        logger.info("正在解析 %s 获取推荐基金代码...", self.report_file)
        if not os.path.exists(self.report_file):
            logger.error("报告文件 %s 不存在", self.report_file)
            raise FileNotFoundError(f"{self.report_file} 不存在")
        
        try:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info("analysis_report.md 内容（前1000字符）: %s", content[:1000])
            
            pattern = re.compile(r'(?:^\| +(\d{6})|### 基金 (\d{6}))', re.M)
            matches = pattern.findall(content)

            extracted_codes = set()
            for match in matches:
                code = match[0] if match[0] else match[1]
                extracted_codes.add(code)
            
            sorted_codes = sorted(list(extracted_codes))
            self.fund_codes = sorted_codes[:10]
            
            if not self.fund_codes:
                logger.warning("未提取到任何有效基金代码，请检查 analysis_report.md")
            else:
                logger.info("提取到 %d 个基金（测试限制前10个）: %s", len(self.fund_codes), self.fund_codes)
            
        except Exception as e:
            logger.error("解析报告文件失败: %s", e)
            raise

    def _get_latest_local_date(self, fund_code):
        """检查本地是否存在数据，并返回最新日期"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    latest_date = df['date'].max().date()
                    logger.info("本地已存在基金 %s 数据，最新日期为: %s", fund_code, latest_date)
                    return latest_date, df
            except Exception as e:
                logger.warning("读取本地文件 %s 失败: %s", file_path, e)
        return None, pd.DataFrame()

    def _save_to_local_file(self, fund_code, new_data_df):
        """将新数据追加到本地文件"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if os.path.exists(file_path):
            new_data_df.to_csv(file_path, mode='a', header=False, index=False)
            logger.info("新数据已追加到本地文件: %s", file_path)
        else:
            new_data_df.to_csv(file_path, index=False)
            logger.info("新数据已保存到本地文件: %s", file_path)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(5),
        retry=tenacity.retry_if_exception_type((requests.RequestException, ValueError)),
        before_sleep=lambda retry_state: logger.info(f"重试基金 {retry_state.args[1]}，第 {retry_state.attempt_number} 次")
    )
    def _get_fund_data_from_eastmoney(self, fund_code):
        """从天天基金API获取基金历史净值数据"""
        logger.info("正在获取基金 %s 的净值数据...", fund_code)
        
        latest_local_date, local_df = self._get_latest_local_date(fund_code)
        start_date = (latest_local_date + timedelta(days=1)).strftime('%Y-%m-%d') if latest_local_date else '1900-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page=1&per=50000&sdate={start_date}&edate={end_date}&rt={int(time.time())}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # 解析表格数据
            content = response.text
            if '<table' not in content:
                logger.warning("基金 %s 返回数据无表格，可能无新数据", fund_code)
                if not local_df.empty:
                    return local_df[['date', 'net_value']].tail(100)
                raise ValueError("无有效数据")
            
            # 使用pandas直接解析HTML表格
            df = pd.read_html(StringIO(content), flavor='lxml')[0]
            df = df[['净值日期', '单位净值']].rename(columns={'净值日期': 'date', '单位净值': 'net_value'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
            df = df.dropna(subset=['date', 'net_value']).sort_values('date').drop_duplicates('date')
            
            if df.empty:
                logger.warning("基金 %s 数据清洗后为空", fund_code)
                if not local_df.empty:
                    return local_df[['date', 'net_value']].tail(100)
                raise ValueError("无有效数据")
            
            # 合并本地和新数据
            if not local_df.empty:
                df_combined = pd.concat([local_df, df]).drop_duplicates(subset=['date']).sort_values(by='date', ascending=True)
            else:
                df_combined = df
                
            # 保存新数据
            if latest_local_date:
                new_df = df[df['date'].dt.date > latest_local_date]
                if not new_df.empty:
                    self._save_to_local_file(fund_code, new_df)
                    logger.info("基金 %s 新增 %d 行数据", fund_code, len(new_df))
            else:
                self._save_to_local_file(fund_code, df_combined)
                logger.info("基金 %s 保存全量数据 %d 行", fund_code, len(df_combined))
            
            df_combined = df_combined.tail(100)
            logger.info("基金 %s 数据获取成功，最新日期: %s, 最新净值: %.4f", 
                       fund_code, df_combined['date'].iloc[-1].strftime('%Y-%m-%d'), df_combined['net_value'].iloc[-1])
            return df_combined[['date', 'net_value']]
            
        except Exception as e:
            logger.error("获取基金 %s 数据失败: %s", fund_code, str(e))
            if not local_df.empty:
                logger.info("使用本地历史数据返回")
                return local_df[['date', 'net_value']].tail(100)
            raise
        
    def process_fund(self, fund_code):
        """处理单个基金，用于多线程调用"""
        try:
            df = self._get_fund_data_from_eastmoney(fund_code)
            
            if df is not None and not df.empty and len(df) >= 14:
                df = df.sort_values(by='date', ascending=True)
                delta = df['net_value'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                
                ma50 = df['net_value'].rolling(window=min(50, len(df)), min_periods=1).mean()
                
                latest_data = df.iloc[-1]
                latest_net_value = latest_data['net_value']
                latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else np.nan
                latest_ma50 = ma50.iloc[-1]
                latest_ma50_ratio = latest_net_value / latest_ma50 if not pd.isna(latest_ma50) and latest_ma50 != 0 else np.nan

                return {
                    'fund_code': fund_code,
                    'latest_net_value': latest_net_value,
                    'rsi': latest_rsi,
                    'ma_ratio': latest_ma50_ratio
                }
            else:
                logger.warning("基金 %s 数据获取失败或数据不足，跳过计算 (数据行数: %s)", fund_code, len(df) if df is not None else 0)
                return None
        except Exception as e:
            logger.error("处理基金 %s 时发生异常: %s", fund_code, str(e))
            return None

    def get_fund_data(self):
        """使用多线程获取所有基金的数据"""
        logger.info("开始使用多线程获取 %d 个基金的数据...", len(self.fund_codes))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.process_fund, self.fund_codes))

        self.fund_data = {
            result['fund_code']: {
                'latest_net_value': result['latest_net_value'],
                'rsi': result['rsi'],
                'ma_ratio': result['ma_ratio']
            }
            for result in results if result is not None
        }

        if len(self.fund_data) > 0:
            logger.info("所有基金数据处理完成。")
        else:
            logger.error("所有基金数据均获取失败。")

    def generate_report(self):
        """生成市场情绪与技术指标监控报告"""
        logger.info("正在生成市场监控报告...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 市场情绪与技术指标监控报告\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 推荐基金技术指标 (处理基金数: {len(self.fund_codes)})\n")
            f.write("| 基金代码 | 最新净值 | RSI | 净值/MA50 | 投资建议 |\n")
            f.write("|----------|----------|-----|-----------|----------|\n")
            
            if not self.fund_codes:
                f.write("| 无 | 无数据 | - | - | 请检查 analysis_report.md 是否包含有效基金代码 |\n")
            else:
                for fund_code in self.fund_codes:
                    if fund_code in self.fund_data and self.fund_data[fund_code] is not None:
                        data = self.fund_data[fund_code]
                        rsi = data['rsi']
                        ma_ratio = data['ma_ratio']

                        rsi_str = f"{rsi:.2f}" if not np.isnan(rsi) else "N/A"
                        ma_ratio_str = f"{ma_ratio:.2f}" if not np.isnan(ma_ratio) else "N/A"

                        advice = (
                            "等待回调" if not np.isnan(rsi) and rsi > 70 or not np.isnan(ma_ratio) and ma_ratio > 1.2 else
                            "可分批买入" if (np.isnan(rsi) or 30 <= rsi <= 70) and (np.isnan(ma_ratio) or 0.8 <= ma_ratio <= 1.2) else
                            "可加仓" if not np.isnan(rsi) and rsi < 30 else "观察"
                        )
                        f.write(f"| {fund_code} | {data['latest_net_value']:.4f} | {rsi_str} | {ma_ratio_str} | {advice} |\n")
                    else:
                        f.write(f"| {fund_code} | 数据获取失败 | - | - | 观察 |\n")
        
        logger.info("报告生成完成: %s", self.output_file)
        with open(self.output_file, 'r', encoding='utf-8') as f:
            logger.info("market_monitor_report.md 内容: %s", f.read())

if __name__ == "__main__":
    try:
        logger.info("脚本启动")
        monitor = MarketMonitor()
        monitor._parse_report()
        monitor.get_fund_data()
        monitor.generate_report()
        logger.info("脚本执行完成")
    except Exception as e:
        logger.error("脚本运行失败: %s", e)
        raise
