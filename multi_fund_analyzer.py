import pandas as pd
import akshare as ak
from datetime import datetime
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import logging

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

class FundDataFetcher:
    """负责所有数据获取、清洗和缓存管理。"""
    # 将 _web_headers 定义为类变量，确保在任何实例方法中都可访问
    _web_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'http://fund.eastmoney.com/'
    }
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json'):
        self.cache_data = cache_data
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.risk_free_rate = self._get_risk_free_rate()
        

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
        
    def _get_risk_free_rate(self) -> float:
        """获取无风险利率，通常使用中国10年期国债收益率。"""
        try:
            rate_df = ak.bond_zh_us_rate()
            # 找到最新的中国10年期国债收益率
            rate = rate_df[rate_df['项目'] == '中国10年期国债收益率']['收盘'].iloc[-1]
            self._log(f"akshare 获取无风险利率成功: {rate}")
            return rate / 100.0  # 转换为小数
        except Exception as e:
            self._log(f"akshare 获取无风险利率失败: {e}")
            self._log("尝试使用一个固定的无风险利率 3.0%")
            return 0.03

    def _get_fund_historical_nav(self, fund_code: str, force_update: bool = False):
        """从 akshare 获取基金历史净值数据。"""
        cache_key = f"{fund_code}_historical_nav"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {fund_code} 历史净值数据。")
            return pd.DataFrame(self.cache[cache_key])
        
        try:
            self._log(f"正在获取基金 {fund_code} 历史净值数据...")
            # 尝试获取历史净值
            df = ak.fund_etf_hist_em(fund=fund_code, period="day")
            if df is not None and not df.empty and '单位净值' in df.columns:
                df['净值日期'] = pd.to_datetime(df['净值日期'])
                df.set_index('净值日期', inplace=True)
                self._log(f"akshare 获取基金 {fund_code} 历史净值成功。")
                self.cache[cache_key] = df.to_dict(orient='records')
                self._save_cache()
                return df
            else:
                self._log(f"akshare 获取基金 {fund_code} 历史净值失败: 数据为空或列名不正确")
        except Exception as e:
            self._log(f"akshare 获取基金 {fund_code} 历史净值失败: {e}")
        return None

    def _get_latest_nav_from_web(self, fund_code: str):
        """
        新增方法：从天天基金网网页抓取最新的单位净值和日增长率。
        """
        try:
            url = f"http://fundf10.eastmoney.com/jjjz_{fund_code}.html"
            self._log(f"尝试网页抓取最新净值：{url}")
            r = requests.get(url, headers=self._web_headers, timeout=10)
            r.raise_for_status()
            r.encoding = 'utf-8'
            soup = BeautifulSoup(r.text, 'lxml')
            
            # --- 修改部分，使用更健壮的选择器 ---
            # 新的路径：p标签class为row row1，b标签class为red lar bold
            p_tag = soup.find('p', class_='row row1')
            if p_tag:
                b_tag = p_tag.find('b', class_='red lar bold')
                if b_tag:
                    text_parts = b_tag.text.strip().split('(')
                    if len(text_parts) == 2:
                        nav = float(text_parts[0].strip())
                        growth_rate_text = text_parts[1].strip().replace('%', '').replace(')', '')
                        growth_rate = float(growth_rate_text)
                        
                        self._log(f"网页抓取最新净值成功: {nav}, 日增长率: {growth_rate}%")
                        return {'nav': nav, 'daily_growth_rate': growth_rate / 100.0}
            
            self._log("网页抓取最新净值失败: 未找到指定的HTML元素")
            return None
            # --- 修改结束 ---

        except Exception as e:
            self._log(f"网页抓取最新净值失败: {e}")
            return None

    def _get_fund_manager_data(self, fund_code: str, force_update: bool = False):
        """获取基金经理数据。"""
        cache_key = f"{fund_code}_manager"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {fund_code} 经理数据。")
            return self.cache[cache_key]

        try:
            self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
            # 尝试 akshare 抓取
            manager_df = ak.fund_manager_em(fund_code=fund_code)
            if manager_df is not None and not manager_df.empty:
                # 获取最新的基金经理
                latest_manager = manager_df.iloc[0]
                manager_name = latest_manager['姓名']
                manager_start_date = latest_manager['任职起始日期']
                
                # 计算任职年限
                start_date = datetime.strptime(manager_start_date, '%Y-%m-%d')
                days_in_service = (datetime.now() - start_date).days
                years_in_service = days_in_service / 365.25
                
                # 获取任职回报
                return_rate = latest_manager['任职回报']
                
                self._log(f"akshare 获取基金 {fund_code} 经理数据成功。")
                data = {
                    'manager_name': manager_name,
                    'years_in_service': years_in_service,
                    'return_rate': return_rate
                }
                self.cache[cache_key] = data
                self._save_cache()
                return data
            else:
                self._log("akshare 获取基金经理数据失败，尝试网页抓取。")
                # 如果 akshare 失败，则进行网页抓取
                return self._get_fund_manager_data_from_web(fund_code)
                
        except Exception as e:
            self._log(f"akshare 获取基金 {fund_code} 经理数据失败，尝试网页抓取: {e}")
            return self._get_fund_manager_data_from_web(fund_code)

    def _get_fund_manager_data_from_web(self, fund_code: str):
        """
        新增方法：从天天基金网网页抓取基金经理数据。
        """
        try:
            url = f"http://fundf10.eastmoney.com/jbgk_{fund_code}.html"
            self._log(f"尝试网页抓取基金经理数据：{url}")
            r = requests.get(url, headers=self._web_headers, timeout=10)
            r.raise_for_status()
            r.encoding = 'utf-8'
            soup = BeautifulSoup(r.text, 'lxml')

            # --- 修改部分，使用更健壮的选择器 ---
            # 新的路径：div class为bs_gl，p标签内部包含'基金经理'的label
            bs_gl_div = soup.find('div', class_='bs_gl')
            if bs_gl_div:
                manager_label = bs_gl_div.find('label', string=re.compile(r'基金经理：'))
                if manager_label:
                    manager_link = manager_label.find('a')
                    if manager_link:
                        manager_name = manager_link.text.strip()
                        self._log(f"网页抓取基金经理姓名成功: {manager_name}")
                        return {'manager_name': manager_name, 'years_in_service': None, 'return_rate': None}
            
            self._log("网页抓取基金经理数据失败: 未找到指定的HTML元素")
            return None
            # --- 修改结束 ---
        
        except Exception as e:
            self._log(f"网页抓取基金经理数据失败: {e}")
            return None

    def _get_fund_holdings_data(self, fund_code: str, force_update: bool = False):
        """
        获取基金持仓数据，包括前十大重仓股和行业配置。
        """
        cache_key = f"{fund_code}_holdings"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {fund_code} 持仓数据。")
            return self.cache[cache_key]
        
        try:
            # --- 修改部分，纠正 URL，使用正确的基金持仓页面 ---
            # 之前的 URL 是 jj_code.html，现在修改为 ccmx_jj_code.html
            url = f"http://fundf10.eastmoney.com/ccmx_{fund_code}.html"
            self._log(f"正在获取基金 {fund_code} 持仓数据: {url}")
            r = requests.get(url, headers=self._web_headers, timeout=10)
            r.raise_for_status()
            r.encoding = 'utf-8'
            soup = BeautifulSoup(r.text, 'lxml')
            
            holdings_table = soup.find('div', id='cctable')
            if holdings_table:
                # 寻找表格
                table = holdings_table.find('table')
                if table:
                    # 解析表格内容
                    holdings_list = []
                    rows = table.find_all('tr')[1:] # 跳过表头
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 5:
                            stock_code = cols[1].find('a').text.strip()
                            stock_name = cols[2].find('a').text.strip()
                            holding_ratio = float(cols[4].text.replace('%', '')) / 100
                            holdings_list.append({'code': stock_code, 'name': stock_name, 'ratio': holding_ratio})
                    
                    self._log(f"获取基金 {fund_code} 持仓数据成功。")
                    self.cache[cache_key] = {'holdings': holdings_list}
                    self._save_cache()
                    return {'holdings': holdings_list}
            
            self._log(f"获取基金 {fund_code} 持仓数据失败: 未找到持仓表格。")
            return None
            # --- 修改结束 ---
            
        except Exception as e:
            self._log(f"获取基金 {fund_code} 持仓数据失败: {e}")
            return None
    
    def _get_fund_info(self, fund_code: str, force_update: bool = False):
        """获取基金规模、成立日期和类型。"""
        cache_key = f"{fund_code}_info"
        if not force_update and cache_key in self.cache:
            self._log(f"使用缓存的基金 {fund_code} 信息。")
            return self.cache[cache_key]
            
        try:
            self._log(f"正在获取基金 {fund_code} 基本信息...")
            # 东方财富基金档案页面
            url = f"http://fundf10.eastmoney.com/jbgk_{fund_code}.html"
            r = requests.get(url, headers=self._web_headers, timeout=10)
            r.raise_for_status()
            r.encoding = 'utf-8'
            soup = BeautifulSoup(r.text, 'lxml')
            
            # 提取信息
            size = None
            size_label = soup.find('label', string=re.compile(r'资产规模：'))
            if size_label:
                size_text = size_label.find_parent('p').text
                size_match = re.search(r'(\d+\.?\d*)亿元', size_text)
                if size_match:
                    size = float(size_match.group(1))

            establish_date = None
            establish_label = soup.find('label', string=re.compile(r'成立日期：'))
            if establish_label:
                date_text = establish_label.find('span').text.strip()
                establish_date = datetime.strptime(date_text, '%Y-%m-%d')
            
            fund_type = None
            type_label = soup.find('label', string=re.compile(r'类型：'))
            if type_label:
                fund_type = type_label.find('span').text.strip()
                
            info_data = {
                'scale': size,
                'establish_date': establish_date.strftime('%Y-%m-%d') if establish_date else None,
                'fund_type': fund_type
            }
            
            self._log(f"获取基金 {fund_code} 信息成功。")
            self.cache[cache_key] = info_data
            self._save_cache()
            return info_data
            
        except Exception as e:
            self._log(f"获取基金 {fund_code} 信息失败: {e}")
            return None

    def _get_market_sentiment(self) -> dict:
        """获取市场情绪，通过分析上证指数。"""
        try:
            self._log("正在获取市场情绪数据...")
            # 使用 akshare 获取上证指数历史数据
            sh_index = ak.stock_zh_index_hist_sh("000001", "20230101", datetime.now().strftime("%Y%m%d"))
            sh_index.set_index('日期', inplace=True)
            
            # 计算指数近一年的涨跌幅
            one_year_ago = datetime.now() - pd.Timedelta(days=365)
            last_year_data = sh_index.loc[sh_index.index > one_year_ago.strftime('%Y-%m-%d')]
            
            if len(last_year_data) < 2:
                self._log("上证指数数据不足，无法判断市场情绪。")
                return {'sentiment': 'unknown', 'trend': 'unknown'}
            
            start_price = last_year_data['开盘'].iloc[0]
            end_price = last_year_data['收盘'].iloc[-1]
            growth_rate = (end_price - start_price) / start_price
            
            # 判断市场趋势
            if growth_rate > 0.1:  # 涨幅超过10%视为牛市
                sentiment = 'bullish'
                trend = 'up'
            elif growth_rate < -0.1: # 跌幅超过10%视为熊市
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
        data = {}
        data['nav'] = self._get_fund_historical_nav(fund_code)
        
        # 如果 akshare 获取历史净值失败，尝试网页抓取最新净值作为补充
        if data['nav'] is None:
            latest_nav_data = self._get_latest_nav_from_web(fund_code)
            if latest_nav_data:
                data['latest_nav'] = latest_nav_data['nav']
                data['daily_growth_rate'] = latest_nav_data['daily_growth_rate']
            else:
                data['latest_nav'] = None
                data['daily_growth_rate'] = None
        
        data['manager'] = self._get_fund_manager_data(fund_code)
        data['holdings'] = self._get_fund_holdings_data(fund_code)
        data['info'] = self._get_fund_info(fund_code)
        data['risk_free_rate'] = self.risk_free_rate
        return data

class InvestmentStrategy:
    """根据数据对基金进行评分和决策。"""
    
    def __init__(self, market_data: dict, logger: logging.Logger):
        self.market_data = market_data
        self._log = lambda msg: logger.info(f"[决策引擎] {msg}")

    def _calculate_sharpe_ratio(self, returns, risk_free_rate):
        """计算夏普比率。"""
        if len(returns) < 2:
            return np.nan
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) # 年化
    
    def _calculate_max_drawdown(self, returns):
        """计算最大回撤。"""
        if len(returns) < 2:
            return np.nan
        nav = (1 + returns).cumprod()
        peak = nav.expanding(min_periods=1).max()
        drawdown = (nav - peak) / peak
        return -drawdown.min()
    
    def _get_benchmark_data(self):
        """获取基准指数数据，用于计算 Beta。"""
        try:
            df = ak.index_hist_pe_pb("000300", "近一年")
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            return df['收盘']
        except Exception as e:
            self._log(f"获取基准指数数据失败: {e}")
            return None
    
    def _calculate_beta(self, fund_returns, benchmark_returns):
        """计算 Beta 系数。"""
        if benchmark_returns is None or fund_returns.empty:
            return np.nan
        
        # 对齐数据
        merged_df = pd.DataFrame({'fund': fund_returns, 'benchmark': benchmark_returns}).dropna()
        if len(merged_df) < 2:
            return np.nan
        
        covariance = merged_df['fund'].cov(merged_df['benchmark'])
        benchmark_variance = merged_df['benchmark'].var()
        
        if benchmark_variance == 0:
            return np.nan
        
        return covariance / benchmark_variance

    def analyze_and_score(self, fund_data: dict, fund_info: dict) -> dict:
        """根据多维度指标对基金进行综合分析和评分。"""
        score = 0
        points_log = {}
        
        fund_nav = fund_data.get('nav')
        manager_data = fund_data.get('manager', {})
        holdings_data = fund_data.get('holdings', {})
        info_data = fund_data.get('info', {})
        
        self._log(f"分析基金类型: {info_data.get('fund_type')}, 规模: {info_data.get('scale')}亿元, 成立年限: {info_data.get('years_since_establish')}年")
        
        # 1. 基金通用指标（40分）
        if fund_nav is not None and not fund_nav.empty:
            returns = fund_nav['日增长率'].astype(float) / 100.0
            
            # 3年涨幅
            rose_3y = fund_nav['累计净值'].iloc[-1] / fund_nav['累计净值'].iloc[0] if len(fund_nav) > 252*3 else np.nan
            points_log['rose_3y'] = 10 if rose_3y > 0.5 else 0
            
            # 3年排名（需要外部数据，此处简化为假设）
            rank_r_3y = 0.1 # 假设排名，此处需真实数据
            points_log['rank_r_3y'] = 10 if rank_r_3y < 0.2 else 0

            # 夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio(returns, fund_data.get('risk_free_rate'))
            points_log['sharpe_ratio'] = 10 if sharpe_ratio > 0.5 else 0
            
            # 最大回撤
            max_drawdown = self._calculate_max_drawdown(returns)
            points_log['max_drawdown'] = 10 if max_drawdown < 0.2 else 0
            
            self._log(f"基金通用指标: 夏普比率: {sharpe_ratio:.4f}, 最大回撤: {max_drawdown:.4f}, 3年涨幅: {rose_3y}, 3年排名: {rank_r_3y}, Beta: N/A")
            
        else:
            self._log("历史数据不足，无法计算性能指标。")
            points_log = {
                'rose_3y': 0, 'rank_r_3y': 0, 
                'sharpe_ratio': np.nan, 'max_drawdown': np.nan,
                'rose_3y_value': np.nan, 'rank_r_3y_value': np.nan,
                'sharpe_ratio_value': np.nan, 'max_drawdown_value': np.nan,
                'beta_value': np.nan
            }

        # 2. 基金经理指标（20分）
        if manager_data and manager_data.get('years_in_service') is not None:
            years = manager_data['years_in_service']
            return_rate = manager_data['return_rate']
            
            points_log['manager_years'] = 10 if years > 3 else 5 if years > 1 else 0
            points_log['manager_return'] = 10 if return_rate > 50 else 5 if return_rate > 20 else 0
            
            self._log(f"基金经理指标: 姓名: {manager_data.get('manager_name')}, 任职年限: {years:.2f}年, 任职回报: {return_rate}%")
        else:
            self._log("基金经理数据不足，无法评分。")
            points_log['manager_years'] = 0
            points_log['manager_return'] = 0
            
        # 3. 基金持仓指标（20分）
        if holdings_data and holdings_data.get('holdings'):
            holdings = holdings_data['holdings']
            top_10_holdings = sorted(holdings, key=lambda x: x['ratio'], reverse=True)[:10]
            top_10_concentration = sum(h['ratio'] for h in top_10_holdings)
            
            points_log['holding_concentration'] = 10 if top_10_concentration < 0.4 else 5 if top_10_concentration < 0.6 else 0
            
            # 行业配置，此处简化为对科技、消费、医疗的偏好
            tech_consumer_healthcare = any('科技' in h['name'] or '消费' in h['name'] or '医疗' in h['name'] for h in top_10_holdings)
            points_log['sector_preference'] = 10 if tech_consumer_healthcare else 5
            
            self._log(f"基金持仓指标: 前十大持仓集中度: {top_10_concentration:.2f}, 行业偏好: {points_log['sector_preference']}")
        else:
            self._log("基金持仓数据不足，无法评分。")
            points_log['holding_concentration'] = 0
            points_log['sector_preference'] = 0
            
        # 4. 基本信息（10分）
        if info_data:
            scale = info_data.get('scale')
            fund_type = info_data.get('fund_type')
            
            if scale and 5 <= scale <= 100:
                points_log['scale'] = 5
            else:
                points_log['scale'] = 0
            
            if fund_type in ['混合型', '股票型']:
                points_log['fund_type'] = 5
            else:
                points_log['fund_type'] = 0
                
            self._log(f"基本信息: 基金类型: {fund_type}, 规模: {scale}亿元")

        # 5. 市场情绪调整（10分）
        market_sentiment = self.market_data.get('sentiment')
        if market_sentiment == 'bullish':
            points_log['market_sentiment_adj'] = 10
        elif market_sentiment == 'neutral':
            points_log['market_sentiment_adj'] = 5
        else:
            points_log['market_sentiment_adj'] = 0
        
        # 计算总分
        for key, value in points_log.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
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
        self.results_df = None

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
        
        for fund_code in fund_codes:
            self._log(f"\n--- 正在分析基金 {fund_code} ---")
            
            fund_data = self.data_fetcher.get_fund_data(fund_code)
            
            fund_info = fund_data.get('info', {})
            if fund_info:
                fund_name = fund_info.get('fund_name', '未知')
            else:
                fund_name = '未知'

            # 评估基金
            analysis_result = self.investment_strategy.analyze_and_score(fund_data, fund_info)
            decision = analysis_result.get('decision')
            score = analysis_result.get('score')
            points_log = analysis_result.get('points_log', {})
            
            # 提取主要指标
            rose_3y = points_log.get('rose_3y_value')
            rank_r_3y = points_log.get('rank_r_3y_value')
            sharpe_ratio = points_log.get('sharpe_ratio_value')
            max_drawdown = points_log.get('max_drawdown_value')
            beta = points_log.get('beta_value')
            scale = fund_info.get('scale')

            results.append({
                'fund_code': fund_code,
                'fund_name': fund_name,
                'rose_3y': rose_3y,
                'rank_r_3y': rank_r_3y,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'beta': beta,
                'scale': scale,
                'decision': decision,
                'score': score
            })
            self._log(f"评分详情: {points_log}")
            time.sleep(1)

        results_df = pd.DataFrame(results)
        
        self._log("\n--- 批量基金分析报告 ---")
        self._log(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"市场趋势: {self.investment_strategy.market_data.get('trend', 'unknown')}")
        
        top_funds = results_df.sort_values('score', ascending=False).head(10)
        if not top_funds.empty:
            self._log("\n--- 推荐基金（综合评分前10名）---")
            self._log("\n" + top_funds[['decision', 'score', 'fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'sharpe_ratio', 'max_drawdown', 'beta', 'scale']].to_markdown(index=False))
        else:
            self._log("\n没有基金获得有效评分。")
        
        self._log("\n" + results_df[['decision', 'score', 'fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'scale']].to_markdown(index=False))
        
        self._save_report_to_markdown()
        
        return results_df

if __name__ == '__main__':
    # 请确保已安装 akshare, pandas, numpy, requests, beautifulsoup4 等库
    # pip install akshare pandas numpy requests beautifulsoup4 lxml
    
    # 基金代码列表 URL
    funds_list_url = 'https://raw.githubusercontent.com/qjlxg/own/main/recommended_cn_funds.csv'
    
    # 获取基金代码列表
    try:
        logger.info("正在从 CSV 导入基金代码列表...")
        # 修复：尝试多种可能的编码格式来读取文件
        encodings = ['utf-8', 'gbk', 'gb18030']
        df_funds = None
        for enc in encodings:
            try:
                df_funds = pd.read_csv(funds_list_url, encoding=enc)
                logger.info(f"成功使用 '{enc}' 编码导入文件。")
                break  # 成功后退出循环
            except UnicodeDecodeError:
                logger.warning(f"使用 '{enc}' 编码失败，尝试下一种。")
        
        if df_funds is not None:
            fund_codes_to_analyze = df_funds['fund_code'].unique().tolist()
            logger.info(f"导入成功，共 {len(fund_codes_to_analyze)} 个基金代码")
        else:
            raise ValueError("所有尝试的编码都无法正确读取文件。")
            
    except Exception as e:
        logger.error(f"导入基金列表失败: {e}")
        fund_codes_to_analyze = []
    
    if fund_codes_to_analyze:
        # 只分析前18个以进行演示和测试
        test_fund_codes = fund_codes_to_analyze[:18] 
        logger.info(f"分析前 {len(test_fund_codes)} 个基金：{test_fund_codes}...")
        
        data_fetcher = FundDataFetcher()
        market_data = data_fetcher._get_market_sentiment()
        
        investment_strategy = InvestmentStrategy(market_data, logger)
        
        analyzer = FundAnalyzer(data_fetcher, investment_strategy)
        analyzer.run_analysis(test_fund_codes)
    else:
        logger.error("无法获取基金代码列表，程序终止。")
