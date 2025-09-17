import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fund_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FundAnalyzer')

class SeleniumFetcher:
    """使用 Selenium 模拟浏览器进行数据抓取。"""
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        try:
            chrome_driver_path = os.getenv('CHROMEDRIVER_PATH', '/usr/local/bin/chromedriver')
            self.driver = webdriver.Chrome(service=ChromeService(chrome_driver_path), options=chrome_options)
            logger.info("Selenium驱动初始化成功。")
        except WebDriverException as e:
            logger.error(f"无法初始化 ChromeDriver: {e}.")
            logger.error("请确保 ChromeDriver 已安装且路径正确。")
            self.driver = None

    def get_page_source(self, url: str) -> str:
        """加载 URL 并返回页面源代码。"""
        if not self.driver:
            logger.error(f"Selenium driver 未初始化，无法访问 {url}")
            return None
        try:
            logger.info(f"使用 Selenium 访问: {url}")
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return self.driver.page_source
        except TimeoutException:
            logger.error(f"加载页面超时: {url}")
            return None
        except Exception as e:
            logger.error(f"Selenium 抓取失败: {e}")
            return None

    def __del__(self):
        """确保浏览器实例在对象销毁时被正确关闭。"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
            logger.info("Selenium驱动已关闭。")

class FundDataFetcher:
    """负责所有数据获取、清洗和缓存管理。"""
    _web_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'http://fund.eastmoney.com/'
    }
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json'):
        self.cache_data = cache_data
        self.cache_file = cache_file
        # --- 修复点：删除缓存文件以确保缓存功能正常工作，这里将注释掉以保留缓存 ---
        # if os.path.exists(self.cache_file):
        #     os.remove(self.cache_file)
        #     logger.info("已清空缓存文件。")
        self.cache = self._load_cache()
        self._selenium_fetcher = None
        self.risk_free_rate = self._get_risk_free_rate()
        self.akshare_rate_limit = 1
        self.last_akshare_call = time.time()

    def _log(self, message: str):
        logger.info(f"[数据获取] {message}")

    def _load_cache(self):
        """加载缓存数据，如果文件不存在则返回空字典。"""
        if self.cache_data and os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    self._log("缓存文件损坏，将重新创建。")
        return {}

    def _save_cache(self):
        """保存当前缓存数据到文件。"""
        if self.cache_data:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
    
    def _pad_fund_code(self, code: str) -> str:
        """确保基金代码为6位字符串，不足则在前面补0。"""
        return code.zfill(6)
        
    @property
    def selenium_fetcher(self):
        """懒加载SeleniumFetcher实例。"""
        if self._selenium_fetcher is None:
            self._log("正在初始化 SeleniumFetcher...")
            self._selenium_fetcher = SeleniumFetcher()
        return self._selenium_fetcher
        
    def _get_risk_free_rate(self) -> float:
        """获取无风险利率，尝试从可靠源获取，失败则用默认值。"""
        url = "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/index.html"
        source = None
        if self.selenium_fetcher.driver:
            source = self.selenium_fetcher.get_page_source(url)
        
        if source:
            soup = BeautifulSoup(source, 'lxml')
            rate_span = soup.find('span', string=re.compile(r'\d+\.\d+%'))
            if rate_span:
                try:
                    rate = float(rate_span.text.strip().replace('%', '')) / 100.0
                    self._log(f"网页抓取无风险利率成功: {rate}")
                    return rate
                except ValueError:
                    self._log("无法解析无风险利率数值。")
        
        self._log("网页抓取无风险利率失败，使用默认值2.5%。")
        return 0.025

    def _wait_for_akshare_call(self):
        """限制 akshare API 调用频率。"""
        elapsed = time.time() - self.last_akshare_call
        if elapsed < self.akshare_rate_limit:
            time.sleep(self.akshare_rate_limit - elapsed)
        self.last_akshare_call = time.time()

    def _get_fund_info(self, fund_code: str, force_update: bool = False) -> dict:
        """从网页抓取基金基本信息（名称、规模、类型）。"""
        padded_code = self._pad_fund_code(fund_code)
        cache_key = f"{padded_code}_info"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {padded_code} 基本信息。")
            return self.cache[cache_key]
        
        url = f"http://fundf10.eastmoney.com/jbgk_{padded_code}.html"
        source = None
        if self.selenium_fetcher.driver:
            source = self.selenium_fetcher.get_page_source(url)
        
        if source:
            soup = BeautifulSoup(source, 'lxml')
            data = {}
            title = soup.find('h4', class_='title')
            if title and title.find('a'):
                data['name'] = title.find('a').text.strip()
            else:
                data['name'] = '未知'
            
            table = soup.find('table', class_='info w790')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    headers = row.find_all('th')
                    if len(cells) >= 2 and len(headers) >= 1:
                        if headers[0].text.strip() == '资产规模':
                            scale_text = cells[0].text.strip()
                            try:
                                scale = float(scale_text.split('亿元')[0])
                                data['scale'] = scale
                            except (ValueError, IndexError):
                                data['scale'] = None
                        if headers[0].text.strip() == '基金类型':
                            data['fund_type'] = cells[0].text.strip()
            
            if data.get('name', '未知') != '未知':
                self.cache[cache_key] = data
                self._save_cache()
                self._log(f"网页抓取基金 {padded_code} 基本信息成功: {data}")
                return data
        self._log(f"网页抓取基金 {padded_code} 基本信息失败。")
        return {'name': '未知', 'scale': None, 'fund_type': '未知'}

    def _get_fund_historical_nav(self, fund_code: str, force_update: bool = False):
        """从 akshare 或网页获取基金历史净值数据。"""
        padded_code = self._pad_fund_code(fund_code)
        cache_key = f"{padded_code}_historical_nav"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {padded_code} 历史净值数据。")
            return pd.DataFrame(self.cache[cache_key])
        
        self._wait_for_akshare_call()
        try:
            self._log(f"正在通过 akshare 获取基金 {padded_code} 历史净值数据...")
            df = ak.fund_open_fund_info_em(fund=padded_code, indicator="unit_net_value")
            if df is not None and not df.empty and '单位净值' in df.columns:
                df['净值日期'] = pd.to_datetime(df['净值日期'])
                df['日增长率'] = df['单位净值'].pct_change().fillna(0)
                df['累计净值'] = df['单位净值']
                df.set_index('净值日期', inplace=True)
                self._log(f"akshare 获取基金 {padded_code} 历史净值成功。")
                self.cache[cache_key] = df.to_dict(orient='records')
                self._save_cache()
                return df
            else:
                self._log(f"akshare 获取基金 {padded_code} 历史净值失败: 数据为空或列名不正确")
        except Exception as e:
            self._log(f"akshare 获取基金 {padded_code} 历史净值失败: {e}")
        
        url = f"http://fundf10.eastmoney.com/jjjz_{padded_code}.html"
        source = None
        if self.selenium_fetcher.driver:
            source = self.selenium_fetcher.get_page_source(url)
        
        if source:
            soup = BeautifulSoup(source, 'lxml')
            table = soup.find('table', class_='w782 comm lsjz')
            if table:
                rows = table.find_all('tr')
                dates, navs, daily_growths = [], [], []
                for row in rows[1:]:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        try:
                            date = pd.to_datetime(cells[0].text.strip())
                            nav = float(cells[1].text.strip())
                            growth_rate = float(cells[2].text.strip().replace('%', '')) / 100.0
                            dates.append(date)
                            navs.append(nav)
                            daily_growths.append(growth_rate)
                        except (ValueError, IndexError):
                            continue
                df = pd.DataFrame({
                    '净值日期': dates,
                    '单位净值': navs,
                    '日增长率': daily_growths,
                    '累计净值': navs
                })
                if not df.empty:
                    df.set_index('净值日期', inplace=True)
                    self.cache[cache_key] = df.to_dict(orient='records')
                    self._save_cache()
                    self._log(f"网页抓取基金 {padded_code} 历史净值成功。")
                    return df
        self._log(f"网页抓取基金 {padded_code} 历史净值失败。")
        return None

    def _get_latest_nav_from_web(self, fund_code: str):
        """从天天基金网网页抓取最新的单位净值和日增长率。"""
        padded_code = self._pad_fund_code(fund_code)
        url = f"http://fundf10.eastmoney.com/jjjz_{padded_code}.html"
        source = None
        if self.selenium_fetcher.driver:
            self.selenium_fetcher.driver.get(url)
            try:
                WebDriverWait(self.selenium_fetcher.driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "row row1"))
                )
                source = self.selenium_fetcher.driver.page_source
            except TimeoutException:
                logger.error(f"加载 {url} 超时")
                return None

        if source:
            soup = BeautifulSoup(source, 'lxml')
            p_tag = soup.find('p', class_='row row1')
            if p_tag:
                b_tag = p_tag.find('b', class_='red lar bold')
                if b_tag:
                    text_parts = b_tag.text.strip().split('(')
                    if len(text_parts) == 2:
                        try:
                            nav = float(text_parts[0].strip())
                            growth_rate_text = text_parts[1].strip().replace('%', '').replace(')', '')
                            growth_rate = float(growth_rate_text)
                            self._log(f"网页抓取最新净值成功: {nav}, 日增长率: {growth_rate}%")
                            return {'nav': nav, 'daily_growth_rate': growth_rate / 100.0}
                        except (ValueError, IndexError):
                            self._log("解析最新净值数据失败。")
        self._log("网页抓取最新净值失败: 未找到指定的HTML元素或数据格式不正确。")
        return None

    def _get_fund_manager_data(self, fund_code: str, force_update: bool = False):
        """获取基金经理数据。"""
        padded_code = self._pad_fund_code(fund_code)
        cache_key = f"{padded_code}_manager"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {padded_code} 经理数据。")
            return self.cache[cache_key]
        
        self._wait_for_akshare_call()
        try:
            self._log(f"正在通过 akshare 获取基金 {padded_code} 的基金经理数据...")
            manager_df = ak.fund_manager_em(fund=padded_code)
            if manager_df is not None and not manager_df.empty:
                latest_manager = manager_df.iloc[0]
                manager_name = latest_manager['姓名']
                manager_start_date = latest_manager['任职起始日期']
                start_date = datetime.strptime(manager_start_date, '%Y-%m-%d')
                years_in_service = (datetime.now() - start_date).days / 365.25
                return_rate = float(latest_manager['任职回报'].replace('%', '')) if isinstance(latest_manager['任职回报'], str) else latest_manager['任职回报']
                
                data = {
                    'manager_name': manager_name,
                    'years_in_service': years_in_service,
                    'return_rate': return_rate
                }
                self._log(f"akshare 获取基金 {padded_code} 经理数据成功。")
                self.cache[cache_key] = data
                self._save_cache()
                return data
            else:
                self._log("akshare 获取基金经理数据失败，尝试网页抓取。")
        except Exception as e:
            self._log(f"akshare 获取基金 {padded_code} 经理数据失败，尝试网页抓取: {e}")
            
        return self._get_fund_manager_data_from_web(padded_code)

    def _get_fund_manager_data_from_web(self, fund_code: str):
        """使用 Selenium 从网页抓取基金经理数据，包括年限和回报率。"""
        url = f"http://fundf10.eastmoney.com/jbgk_{fund_code}.html"
        source = None
        if self.selenium_fetcher.driver:
            self.selenium_fetcher.driver.get(url)
            try:
                WebDriverWait(self.selenium_fetcher.driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "bs_gl"))
                )
                source = self.selenium_fetcher.driver.page_source
            except TimeoutException:
                logger.error(f"加载 {url} 超时")
                return None

        if source:
            soup = BeautifulSoup(source, 'lxml')
            labels = soup.find_all('label')
            manager_name = None
            manager_url = None
            for label in labels:
                if '基金经理' in label.text.strip():
                    manager_link = label.find('a')
                    if manager_link:
                        manager_name = manager_link.text.strip()
                        manager_url = manager_link['href']
                        break
            
            if manager_name and manager_url:
                source_manager = self.selenium_fetcher.get_page_source(manager_url)
                if source_manager:
                    soup_manager = BeautifulSoup(source_manager, 'lxml')
                    table = soup_manager.find('table', class_='fxtb')
                    if table:
                        rows = table.find_all('tr')
                        years_in_service = None
                        return_rate = None
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) > 2:
                                if '任职时间' in cells[0].text.strip():
                                    try:
                                        start_date = pd.to_datetime(cells[1].text.strip().split('至')[0])
                                        years_in_service = (datetime.now() - start_date).days / 365.25
                                    except (ValueError, IndexError):
                                        pass
                                if '任职回报' in cells[0].text.strip():
                                    try:
                                        return_rate = float(cells[1].text.strip().replace('%', ''))
                                    except (ValueError, IndexError):
                                        pass
                        data = {
                            'manager_name': manager_name,
                            'years_in_service': years_in_service,
                            'return_rate': return_rate
                        }
                        self.cache[f"{fund_code}_manager"] = data
                        self._save_cache()
                        logger.info(f"网页抓取基金经理数据成功: {data}")
                        return data
            logger.warning("未找到基金经理标签或经理页面")
        logger.error("网页抓取基金经理数据失败。")
        return None

    def _get_fund_holdings_data(self, fund_code: str, force_update: bool = False):
        """使用 Selenium 获取基金持仓数据，改进表格解析。"""
        padded_code = self._pad_fund_code(fund_code)
        cache_key = f"{padded_code}_holdings"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {padded_code} 持仓数据。")
            return self.cache[cache_key]
        
        url = f"http://fundf10.eastmoney.com/ccmx_{padded_code}.html"
        source = None
        if self.selenium_fetcher.driver:
            self.selenium_fetcher.driver.get(url)
            try:
                WebDriverWait(self.selenium_fetcher.driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "w782"))
                )
                source = self.selenium_fetcher.get_page_source(url)
            except TimeoutException:
                logger.error(f"加载 {url} 超时")
                return None
            
        if source:
            soup = BeautifulSoup(source, 'lxml')
            holdings = []
            table = soup.find('table', class_='w782 comm cc')
            if table:
                rows = table.find_all('tr')
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) > 3:
                        try:
                            holding = {
                                'name': cols[1].text.strip(),
                                'ratio': float(cols[3].text.strip().replace('%', '')) / 100.0 if cols[3].text.strip() else 0.0
                            }
                            holdings.append(holding)
                        except (ValueError, IndexError):
                            continue
            if holdings:
                data = {'holdings': holdings}
                self.cache[cache_key] = data
                self._save_cache()
                self._log(f"网页抓取基金 {padded_code} 持仓数据成功。")
                return data
        self._log(f"网页抓取基金 {padded_code} 持仓数据失败。")
        return None

    def _get_market_sentiment(self) -> dict:
        """获取市场情绪数据，使用上证指数作为基准。"""
        try:
            self._log("正在获取市场情绪数据...")
            df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=(datetime.now() - timedelta(days=365)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
            if df.empty:
                self._log("未能获取上证指数数据，使用默认值。")
                return {'sentiment': 'unknown', 'trend': 'unknown'}
            
            last_year_data = df.tail(252)
            if last_year_data.empty:
                return {'sentiment': 'unknown', 'trend': 'unknown'}
            
            start_price = last_year_data['开盘'].iloc[0]
            end_price = last_year_data['收盘'].iloc[-1]
            growth_rate = (end_price - start_price) / start_price
            
            if growth_rate > 0.1:
                sentiment = 'bullish'
                trend = 'up'
            elif growth_rate < -0.1:
                sentiment = 'bearish'
                trend = 'down'
            else:
                sentiment = 'neutral'
                trend = 'flat'
            
            self._log(f"市场情绪数据已获取：{{'sentiment': '{sentiment}', 'trend': '{trend}'}}")
            return {'sentiment': sentiment, 'trend': trend}
            
        except Exception as e:
            self._log(f"获取市场情绪数据失败: {e}")
            return {'sentiment': 'unknown', 'trend': 'unknown'}

    def get_fund_data(self, fund_code: str):
        """主入口，获取并整合所有基金数据。"""
        try:
            data = {}
            data['info'] = self._get_fund_info(fund_code)
            data['nav'] = self._get_fund_historical_nav(fund_code)
            data['manager'] = self._get_fund_manager_data(fund_code)
            data['holdings'] = self._get_fund_holdings_data(fund_code)
            data['risk_free_rate'] = self.risk_free_rate
            self._log(f"基金 {fund_code} 数据获取完成: {data}")
            return data
        except Exception as e:
            self._log(f"基金 {fund_code} 数据获取失败: {e}")
            return None

class InvestmentStrategy:
    """根据数据对基金进行评分和决策。"""
    def __init__(self, market_data: dict, logger: logging.Logger):
        self.market_data = market_data
        self._log = lambda msg: logger.info(f"[决策引擎] {msg}")

    def _calculate_sharpe_ratio(self, returns, risk_free_rate):
        """计算夏普比率。"""
        if returns.empty or len(returns) < 2:
            return np.nan
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, returns):
        """计算最大回撤。"""
        if returns.empty or len(returns) < 2:
            return np.nan
        nav = (1 + returns).cumprod()
        peak = nav.expanding(min_periods=1).max()
        drawdown = (nav - peak) / peak
        return -drawdown.min()

    def _calculate_beta(self, fund_returns, benchmark_returns):
        """计算 Beta 系数。"""
        if benchmark_returns is None or fund_returns.empty:
            return np.nan
        merged_df = pd.DataFrame({'fund': fund_returns, 'benchmark': benchmark_returns}).dropna()
        if len(merged_df) < 2:
            return np.nan
        covariance = merged_df['fund'].cov(merged_df['benchmark'])
        benchmark_variance = merged_df['benchmark'].var()
        if benchmark_variance == 0:
            return np.nan
        return covariance / benchmark_variance

    def analyze_and_score(self, fund_data: dict) -> dict:
        """根据多维度指标对基金进行综合分析和评分。"""
        score = 0
        points_log = {}
        
        fund_nav = fund_data.get('nav')
        manager_data = fund_data.get('manager', {})
        holdings_data = fund_data.get('holdings', {})
        info_data = fund_data.get('info', {})
        
        fund_type = info_data.get('fund_type', '未知')
        scale = info_data.get('scale', np.nan)
        
        if fund_nav is not None and not fund_nav.empty:
            returns = fund_nav['日增长率'].astype(float)
            
            rose_3y_val = fund_nav['累计净值'].iloc[-1] / fund_nav['累计净值'].iloc[0] - 1 if len(fund_nav) > 252*3 else np.nan
            points_log['rose_3y_score'] = 10 if rose_3y_val > 0.5 else 0
            
            # --- 修复点：移除了随机评分，因为此指标需要同类排名数据，目前无法计算 ---
            # points_log['rank_r_3y_score'] = 10 if np.random.rand() < 0.2 else 0
            points_log['rank_r_3y_score'] = 0
            
            sharpe_ratio_val = self._calculate_sharpe_ratio(returns, fund_data.get('risk_free_rate'))
            points_log['sharpe_ratio_score'] = 10 if sharpe_ratio_val > 0.5 else 0
            
            max_drawdown_val = self._calculate_max_drawdown(returns)
            points_log['max_drawdown_score'] = 10 if max_drawdown_val < 0.2 else 0
            
            self._log(f"通用指标: 夏普比率: {sharpe_ratio_val:.4f}, 最大回撤: {max_drawdown_val:.4f}, 3年涨幅: {rose_3y_val:.2%}")
            
            points_log.update({
                'rose_3y_value': rose_3y_val, 'rank_r_3y_value': np.nan,
                'sharpe_ratio_value': sharpe_ratio_val, 'max_drawdown_value': max_drawdown_val
            })
        else:
            self._log("历史数据不足，无法计算性能指标。")
            points_log.update({
                'rose_3y_score': 0, 'rank_r_3y_score': 0, 'sharpe_ratio_score': 0, 'max_drawdown_score': 0,
                'rose_3y_value': np.nan, 'rank_r_3y_value': np.nan, 'sharpe_ratio_value': np.nan, 'max_drawdown_value': np.nan
            })

        if manager_data and manager_data.get('years_in_service') is not None:
            years = manager_data['years_in_service']
            return_rate = manager_data['return_rate']
            
            points_log['manager_years_score'] = 10 if years > 3 else 5 if years > 1 else 0
            points_log['manager_return_score'] = 10 if return_rate > 50 else 5 if return_rate > 20 else 0
            
            self._log(f"基金经理指标: 任职年限: {years:.2f}年, 任职回报: {return_rate}%")
        else:
            self._log("基金经理数据不足，无法评分。")
            points_log['manager_years_score'] = 0
            points_log['manager_return_score'] = 0
            
        if holdings_data and holdings_data.get('holdings'):
            holdings = holdings_data['holdings']
            top_10_concentration = sum(h['ratio'] for h in holdings[:10]) if len(holdings) >= 10 else 0
            points_log['holding_concentration_score'] = 10 if top_10_concentration < 0.4 else 5 if top_10_concentration < 0.6 else 0
            
            tech_consumer_healthcare = any('科技' in h['name'] or '消费' in h['name'] or '医疗' in h['name'] for h in holdings[:10])
            points_log['sector_preference_score'] = 10 if tech_consumer_healthcare else 5
            
            self._log(f"基金持仓指标: 前十大持仓集中度: {top_10_concentration:.2f}")
            points_log['holding_concentration_value'] = top_10_concentration
        else:
            self._log("基金持仓数据不足，无法评分。")
            points_log['holding_concentration_score'] = 0
            points_log['sector_preference_score'] = 0
            points_log['holding_concentration_value'] = np.nan
            
        if info_data:
            if scale and 5 <= scale <= 100:
                points_log['scale_score'] = 5
            else:
                points_log['scale_score'] = 0
            
            if fund_type in ['混合型-灵活', '股票型']:
                points_log['fund_type_score'] = 5
            else:
                points_log['fund_type_score'] = 0
                
            self._log(f"基本信息: 基金类型: {fund_type}, 规模: {scale}亿元")

        market_sentiment = self.market_data.get('sentiment', 'unknown')
        if market_sentiment == 'bullish':
            points_log['market_sentiment_adj_score'] = 10
        elif market_sentiment == 'neutral':
            points_log['market_sentiment_adj_score'] = 5
        else:
            points_log['market_sentiment_adj_score'] = 0
        
        for key, value in points_log.items():
            if key.endswith('_score') and isinstance(value, (int, float)) and not np.isnan(value):
                score += value

        decision = "观望"
        if score > 80:
            decision = "强烈推荐"
        elif score > 60:
            decision = "建议持有"
        elif score > 40:
            decision = "评估其他基金"

        return {
            'decision': decision,
            'score': score,
            'points_log': points_log
        }
    
class FundAnalyzer:
    """主程序，协调数据获取和投资策略。"""
    
    def __init__(self, data_fetcher: FundDataFetcher, investment_strategy: InvestmentStrategy):
        self.data_fetcher = data_fetcher
        self.investment_strategy = investment_strategy
        self.report_content = ""

    def _log(self, message: str):
        logger.info(f"[分析报告] {message}")
        self.report_content += message + "\n"

    def _save_report_to_markdown(self):
        """将报告内容保存为Markdown文件。"""
        with open('analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(self.report_content)

    def run_analysis(self, fund_codes: list):
        """执行批量基金分析。"""
        results = []
        
        self._log("--- 批量基金分析启动 ---")
        
        # --- 修复点：将调试代码注释掉，以分析所有基金 ---
        # fund_codes = fund_codes[:1]
        self._log(f"分析基金：{fund_codes}")
        
        for fund_code in fund_codes:
            self._log(f"\n--- 正在分析基金 {fund_code} ---")
            
            try:
                fund_data = self.data_fetcher.get_fund_data(fund_code)
                
                if not fund_data or not fund_data.get('info'):
                    self._log(f"基金 {fund_code} 基本信息获取失败，跳过分析。")
                    continue
                
                fund_info = fund_data['info']
                fund_name = fund_info.get('name', '未知')
                
                analysis_result = self.investment_strategy.analyze_and_score(fund_data)
                decision = analysis_result.get('decision')
                score = analysis_result.get('score')
                points_log = analysis_result.get('points_log', {})
                
                results.append({
                    'fund_code': fund_code,
                    'fund_name': fund_name,
                    'rose_3y': f"{points_log.get('rose_3y_value', np.nan):.2%}" if not np.isnan(points_log.get('rose_3y_value', np.nan)) else np.nan,
                    'sharpe_ratio': f"{points_log.get('sharpe_ratio_value', np.nan):.4f}" if not np.isnan(points_log.get('sharpe_ratio_value', np.nan)) else np.nan,
                    'max_drawdown': f"{points_log.get('max_drawdown_value', np.nan):.2%}" if not np.isnan(points_log.get('max_drawdown_value', np.nan)) else np.nan,
                    'scale': fund_info.get('scale'),
                    'decision': decision,
                    'score': score
                })
                self._log(f"评分详情: {points_log}")
            except Exception as e:
                self._log(f"分析基金 {fund_code} 失败: {e}")
            time.sleep(1)

        results_df = pd.DataFrame(results)
        
        self._log("\n--- 批量基金分析报告 ---")
        self._log(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"市场趋势: {self.investment_strategy.market_data.get('trend', 'unknown')}")
        
        if not results_df.empty:
            top_funds = results_df.sort_values('score', ascending=False).head(10)
            self._log("\n--- 推荐基金（综合评分前10名）---")
            self._log("\n" + top_funds[['decision', 'score', 'fund_code', 'fund_name', 'rose_3y', 'sharpe_ratio', 'max_drawdown', 'scale']].to_markdown(index=False))
            self._log("\n--- 全部基金分析结果 ---")
            self._log("\n" + results_df[['decision', 'score', 'fund_code', 'fund_name', 'scale']].to_markdown(index=False))
        else:
            self._log("\n没有基金获得有效评分。")
        
        self._save_report_to_markdown()
        
        return results_df

if __name__ == '__main__':
    funds_list_url = 'https://raw.githubusercontent.com/qjlxg/own/main/recommended_cn_funds.csv'
    
    try:
        logger.info("正在从 CSV 导入基金代码列表...")
        df_funds = pd.read_csv(funds_list_url, encoding='gb18030')
        fund_codes_to_analyze = [str(code).zfill(6) for code in df_funds['code'].unique().tolist()]
        logger.info(f"导入成功，共 {len(fund_codes_to_analyze)} 个基金代码")
    except Exception as e:
        logger.error(f"导入基金列表失败: {e}")
        fund_codes_to_analyze = ['000363']
    
    if fund_codes_to_analyze:
        logger.info(f"分析基金：{fund_codes_to_analyze}...")
        data_fetcher = FundDataFetcher()
        market_data = data_fetcher._get_market_sentiment()
        investment_strategy = InvestmentStrategy(market_data, logger)
        analyzer = FundAnalyzer(data_fetcher, investment_strategy)
        analyzer.run_analysis(fund_codes_to_analyze)
    else:
        logger.error("无法获取基金代码列表，程序终止。")
