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
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json'):
        self.cache_data = cache_data
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.risk_free_rate = self._get_risk_free_rate()
        self._web_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    def _log(self, message: str):
        logger.info(f"[数据获取] {message}")

    def _load_cache(self) -> dict:
        """从文件加载缓存并检查时效性。"""
        if not os.path.exists(self.cache_file):
            self._log("缓存文件不存在，将创建新缓存。")
            return {}
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            if 'last_updated' not in cache:
                self._log("缓存文件缺少 'last_updated' 键，将重新生成缓存。")
                return {}
            
            last_updated = datetime.strptime(cache.get('last_updated'), '%Y-%m-%d')
            if (datetime.now() - last_updated).days > 1:
                self._log("缓存数据已过期，将重新获取。")
                return {}
            
            self._log("使用有效缓存数据。")
            return cache
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            self._log(f"缓存文件加载失败或格式错误: {e}。将创建新缓存。")
            return {}

    def _save_cache(self):
        """将缓存保存到文件。"""
        self.cache['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)
        self._log("缓存数据已保存。")

    def _get_risk_free_rate(self) -> float:
        """从东方财富获取最新 10 年期国债收益率。"""
        try:
            bond_data = ak.bond_zh_us_rate()
            risk_free_rate = bond_data[bond_data['item_name'] == '中国10年期国债']['value'].iloc[-1] / 100
            self._log(f"获取最新无风险利率：{risk_free_rate:.4f}")
            return risk_free_rate
        except Exception as e:
            self._log(f"获取无风险利率失败，使用默认值 0.018298: {e}")
            return 0.018298
    
    def get_market_sentiment(self):
        """获取市场情绪（基于上证指数）。"""
        if 'market' in self.cache:
            self._log("使用缓存的市场情绪数据。")
            return self.cache['market']
        
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
            
            market_data = {'sentiment': sentiment, 'trend': trend}
            self.cache['market'] = market_data
            self._save_cache()
            self._log(f"市场情绪数据已获取：{market_data}")
            return market_data
        except Exception as e:
            self._log(f"获取市场数据失败: {e}")
            return {'sentiment': 'unknown', 'trend': 'unknown'}
            
    def get_real_time_fund_data(self, fund_code: str, fund_name: str) -> dict:
        """获取单个基金的实时数据和性能指标。"""
        cache_key = f"{fund_code}_{fund_name}"
        if self.cache_data and cache_key in self.cache.get('fund_metrics', {}):
            self._log(f"使用缓存的基金 {fund_code} 数据。")
            return self.cache['fund_metrics'][cache_key]

        self._log(f"正在获取基金 {fund_code} ({fund_name}) 的实时数据和性能指标...")
        for attempt in range(3):
            try:
                fund_data_ak = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
                fund_data_ak['净值日期'] = pd.to_datetime(fund_data_ak['净值日期'])
                fund_data_ak.set_index('净值日期', inplace=True)
                fund_data_ak = fund_data_ak.dropna()

                if len(fund_data_ak) < 252:
                    raise ValueError("数据不足，无法计算可靠的指标")

                returns = fund_data_ak['单位净值'].pct_change().dropna()
                annual_returns_calc = returns.mean() * 252
                annual_volatility_calc = returns.std() * (252**0.5)
                sharpe_ratio = (annual_returns_calc - self.risk_free_rate) / annual_volatility_calc if annual_volatility_calc != 0 else 0
                
                rolling_max = fund_data_ak['单位净值'].cummax()
                daily_drawdown = (fund_data_ak['单位净值'] - rolling_max) / rolling_max
                max_drawdown = daily_drawdown.min() * -1
                
                data = {
                    'latest_nav': float(fund_data_ak['单位净值'].iloc[-1]),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'tracking_error': np.nan
                }
                
                if self.cache_data:
                    self.cache.setdefault('fund_metrics', {})[cache_key] = data
                    self._save_cache()
                self._log(f"基金 {fund_code} 数据获取成功：{data}")
                return data
            except Exception as e:
                self._log(f"获取基金 {fund_code} 数据失败 (尝试 {attempt+1}/3): {e}")
                time.sleep(2)
        self._log(f"基金 {fund_code} 数据获取失败，返回默认值。")
        return {
            'latest_nav': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'tracking_error': np.nan
        }

    def get_fund_manager_data(self, fund_code: str) -> dict:
        """获取基金经理数据。"""
        if self.cache_data and fund_code in self.cache.get('manager', {}):
            self._log(f"使用缓存的基金 {fund_code} 经理数据。")
            return self.cache['manager'][fund_code]
        
        self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
        manager_data = self._scrape_manager_data_from_web(fund_code)
        if self.cache_data:
            self.cache.setdefault('manager', {})[fund_code] = manager_data
            self._save_cache()
        self._log(f"基金 {fund_code} 经理数据获取成功：{manager_data}")
        return manager_data

    def _scrape_manager_data_from_web(self, fund_code: str) -> dict:
        """从天天基金网通过网页抓取获取基金经理数据。"""
        manager_url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        try:
            response = requests.get(manager_url, headers=self._web_headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('label', string='基金经理变动一览').find_parent().find_next_sibling('table')
            if not table or len(table.find_all('tr')) < 2:
                self._log(f"基金 {fund_code} 经理数据表格未找到或数据不足。")
                return {}
            
            latest_manager_row = table.find_all('tr')[1]
            cols = latest_manager_row.find_all('td')
            if len(cols) < 5:
                self._log(f"基金 {fund_code} 经理数据列不足。")
                return {}

            manager_name = cols[2].text.strip()
            tenure_str = cols[3].text.strip()
            cumulative_return_str = cols[4].text.strip()
            
            tenure_days = float(re.search(r'\d+', tenure_str).group()) if '天' in tenure_str else (
                float(re.search(r'\d+', tenure_str).group()) * 365 if '年' in tenure_str else np.nan)
            
            cumulative_return = float(re.search(r'[-+]?\d*\.?\d+', cumulative_return_str).group()) if '%' in cumulative_return_str else np.nan
            
            return {
                'name': manager_name,
                'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                'cumulative_return': cumulative_return
            }
        except Exception as e:
            self._log(f"网页抓取基金 {fund_code} 经理数据失败: {e}")
            return {}

    def get_fund_holdings_data(self, fund_code: str) -> dict:
        """获取基金的股票持仓和行业配置数据。"""
        if self.cache_data and fund_code in self.cache.get('holdings', {}):
            self._log(f"使用缓存的基金 {fund_code} 持仓数据。")
            return self.cache['holdings'][fund_code]

        self._log(f"正在获取基金 {fund_code} 的持仓数据...")
        holdings_url = f"http://fundf10.eastmoney.com/ccmx_{fund_code}.html"
        
        try:
            response = requests.get(holdings_url, headers=self._web_headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            holdings = []
            holdings_header = soup.find('h4', string=lambda t: t and '股票投资明细' in t)
            if holdings_header:
                holdings_table = holdings_header.find_next('table')
                if holdings_table:
                    rows = holdings_table.find_all('tr')[1:]
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 5:
                            holdings.append({
                                '股票代码': cols[1].text.strip(),
                                '股票名称': cols[2].text.strip(),
                                '占净值比例': float(cols[4].text.strip().replace('%', '')),
                            })
            
            sectors = []
            sector_header = soup.find('h4', string=lambda t: t and '股票行业配置' in t)
            if sector_header:
                sector_table = sector_header.find_next('table')
                if sector_table:
                    rows = sector_table.find_all('tr')[1:]
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            sectors.append({
                                '行业名称': cols[0].text.strip(),
                                '占净值比例': float(cols[1].text.strip().replace('%', '')),
                            })

            data = {'holdings': holdings, 'sectors': sectors}
            if self.cache_data:
                self.cache.setdefault('holdings', {})[fund_code] = data
                self._save_cache()
            self._log(f"基金 {fund_code} 持仓数据获取成功：持仓数 {len(holdings)}, 行业数 {len(sectors)}")
            return data
        except Exception as e:
            self._log(f"获取基金 {fund_code} 持仓数据失败: {e}")
            return {'holdings': [], 'sectors': []}

    def get_fund_info(self, fund_code: str) -> dict:
        """获取基金规模和成立年限。"""
        cache_key = f"{fund_code}_info"
        if self.cache_data and cache_key in self.cache.get('fund_info', {}):
            self._log(f"使用缓存的基金 {fund_code} 信息。")
            return self.cache['fund_info'][cache_key]

        self._log(f"正在获取基金 {fund_code} 的规模和成立年限...")
        try:
            fund_info = ak.fund_open_fund_info_em(symbol=fund_code, indicator="基金概况")
            scale = float(fund_info.get('基金规模', np.nan)) if pd.notna(fund_info.get('基金规模')) else np.nan
            establishment_date = pd.to_datetime(fund_info.get('成立日期', np.nan))
            years_since_establishment = (datetime.now() - establishment_date).days / 365.0 if pd.notna(establishment_date) else np.nan
            
            data = {
                'scale': scale,  # 基金规模（亿元）
                'years_since_establishment': years_since_establishment
            }
            
            if self.cache_data:
                self.cache.setdefault('fund_info', {})[cache_key] = data
                self._save_cache()
            self._log(f"基金 {fund_code} 信息获取成功：{data}")
            return data
        except Exception as e:
            self._log(f"获取基金 {fund_code} 信息失败: {e}")
            return {'scale': np.nan, 'years_since_establishment': np.nan}

    def get_index_valuation(self, fund_code: str) -> dict:
        """从理杏仁网站获取指数基金的估值数据（模拟低估值判断）。"""
        cache_key = f"{fund_code}_valuation"
        if self.cache_data and cache_key in self.cache.get('valuation', {}):
            self._log(f"使用缓存的基金 {fund_code} 估值数据。")
            return self.cache['valuation'][cache_key]

        self._log(f"正在获取基金 {fund_code} 的估值数据...")
        try:
            # 模拟估值数据（实际需替换为理杏仁API或网页抓取）
            valuation_data = {
                'pe': np.random.uniform(10, 20),  # 模拟市盈率
                'pb': np.random.uniform(1, 2),    # 模拟市净率
                'is_low_valuation': False
            }
            if valuation_data['pe'] < 15 or valuation_data['pb'] < 1.5:
                valuation_data['is_low_valuation'] = True

            if self.cache_data:
                self.cache.setdefault('valuation', {})[cache_key] = valuation_data
                self._save_cache()
            self._log(f"基金 {fund_code} 估值数据获取成功：{valuation_data}")
            return valuation_data
        except Exception as e:
            self._log(f"获取基金 {fund_code} 估值数据失败: {e}")
            return {'pe': np.nan, 'pb': np.nan, 'is_low_valuation': False}

class InvestmentStrategy:
    """封装所有投资决策逻辑，融入基金.md的筛选方法。"""
    def __init__(self, market_data: dict, personal_strategy: dict):
        self.market_data = market_data
        self.personal_strategy = personal_strategy
        self.points_log = {}
        self.hot_sectors = ['新能源', '军工', '银行', '金融地产']  # 热门板块

    def _log(self, message: str):
        logger.info(f"[决策引擎] {message}")

    def score_fund(self, fund_data: dict, fund_info: dict, manager_data: dict, holdings_data: dict, fund_metadata: dict) -> tuple[float, dict]:
        """
        基于多维指标为基金打分，融入基金.md的筛选方法。
        分数越高，推荐度越高。
        """
        score = 0
        self.points_log = {}
        market_trend = self.market_data.get('trend', 'neutral')
        fund_type = fund_info.get('类型', '未知')
        scale = fund_metadata.get('scale', np.nan)
        years_since_establishment = fund_metadata.get('years_since_establishment', np.nan)
        
        self._log(f"分析基金类型: {fund_type}, 规模: {scale}亿元, 成立年限: {years_since_establishment}年")

        # 1. 基金类型偏好（混合、债券、指数优先）
        if fund_type in ['混合型', '债券型', '指数型']:
            score += 10
            self.points_log[f'基金类型为{fund_type}'] = 10
        elif fund_type == '货币型' and not ('A' in fund_info.get('name', '') or 'B' in fund_info.get('name', '')):
            score += 5
            self.points_log['货币型基金（无A/B类）'] = 5

        # 2. 基金规模和成立年限
        if pd.notna(scale):
            if fund_type == '股票型' and scale > 10:
                score += 10
                self.points_log['股票型基金规模 > 10亿'] = 10
            elif fund_type == '指数型' and scale > 1:
                score += 10
                self.points_log['指数型基金规模 > 1亿'] = 10
        if pd.notna(years_since_establishment) and years_since_establishment > 5 and fund_type == '货币型':
            score += 5
            self.points_log['货币型基金成立年限 > 5年'] = 5

        # 3. 基金通用评分（基于净值数据）
        sharpe_ratio = fund_data.get('sharpe_ratio')
        max_drawdown = fund_data.get('max_drawdown')
        rose_3y = fund_info.get('rose_3y', np.nan)
        rank_r_3y = fund_info.get('rank_r_3y', np.nan)
        
        sharpe_ratio_str = f"{sharpe_ratio:.4f}" if pd.notna(sharpe_ratio) else "N/A"
        max_drawdown_str = f"{max_drawdown:.4f}" if pd.notna(max_drawdown) else "N/A"
        self._log(f"基金通用指标: 夏普比率: {sharpe_ratio_str}, 最大回撤: {max_drawdown_str}, 3年涨幅: {rose_3y}, 3年排名: {rank_r_3y}")
        
        # 收益与风险评分（强化股票和混合基金）
        if pd.notna(rose_3y):
            if rose_3y > 100:
                score += 20
                self.points_log['3年涨幅 > 100%'] = 20
            elif rose_3y > 50:
                score += 10
                self.points_log['3年涨幅 > 50%'] = 10
        if pd.notna(rank_r_3y) and rank_r_3y < 0.05:
            score += 15
            self.points_log['3年排名 < 5%'] = 15
        if pd.notna(sharpe_ratio):
            if sharpe_ratio > 1.0 and fund_type in ['股票型', '混合型']:
                score += 20
                self.points_log['夏普比率 > 1.0'] = 20
            elif sharpe_ratio > 0.5:
                score += 10
                self.points_log['夏普比率 > 0.5'] = 10
        if pd.notna(max_drawdown) and max_drawdown < 0.2 and fund_type in ['股票型', '混合型']:
            score += 15
            self.points_log['最大回撤 < 20%'] = 15

        # 4. 基金经理评分（三年内无变动）
        manager_name = manager_data.get('name', 'N/A')
        tenure_years = manager_data.get('tenure_years')
        manager_return = manager_data.get('cumulative_return')
        self._log(f"基金经理指标: 姓名: {manager_name}, 任职年限: {tenure_years}年, 任职回报: {manager_return}%")
        if pd.notna(tenure_years):
            if tenure_years > 3 and pd.notna(manager_return) and manager_return > 20:
                score += 20
                self.points_log['基金经理资深且回报高'] = 20
            elif tenure_years < 3 and fund_type in ['债券型', '混合型']:
                score -= 10
                self.points_log['基金经理三年内变动'] = -10

        # 5. 持仓评分（热门板块和集中度）
        holdings = holdings_data.get('holdings', [])
        sectors = holdings_data.get('sectors', [])
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            top_10_concentration = holdings_df['占净值比例'].iloc[:10].sum() if len(holdings_df) >= 10 else holdings_df['占净值比例'].sum()
            if top_10_concentration > 60:
                score -= 15
                self.points_log['前十持仓集中度 > 60%'] = -15
        
        # 热门板块加分
        if sectors:
            sectors_df = pd.DataFrame(sectors)
            hot_sector_weight = sum(s['占净值比例'] for s in sectors if any(hs in s['行业名称'] for hs in self.hot_sectors))
            if hot_sector_weight > 20:
                score += 10
                self.points_log['热门板块占比 > 20%'] = 10

        # 6. 估值评分（指数基金）
        if fund_type == '指数型' and fund_metadata.get('is_low_valuation', False):
            score += 15
            self.points_log['指数基金低估值'] = 15

        # 7. 市场趋势调整
        original_score = score
        if market_trend == 'bullish':
            score *= 1.1
            self.points_log['市场趋势: 牛市'] = score - original_score
        elif market_trend == 'bearish':
            score *= 0.9
            self.points_log['市场趋势: 熊市'] = score - original_score
        
        self._log(f"基金评分完成，总分: {score:.2f}, 得分详情: {self.points_log}")
        return score, self.points_log

    def make_decision(self, score: float) -> str:
        """根据最终得分给出投资建议。"""
        if score > 80:
            return "强烈推荐：综合评分极高，适合作为核心持仓。"
        elif score > 50:
            return "建议持有或加仓：表现良好，可作为卫星持仓。"
        elif score > 20:
            return "观望：表现一般，需要持续观察。"
        else:
            return "评估其他基金：综合评分较低，存在明显不足。"

class FundAnalyzer:
    """主程序，协调数据获取和投资决策并生成报告。"""
    def __init__(self, cache_data: bool = True):
        self.data_fetcher = FundDataFetcher(cache_data=cache_data)
        self.analysis_report = []

    def _log(self, message: str):
        """记录分析报告到列表，稍后写入 Markdown 文件"""
        print(f"[分析报告] {message}")
        self.analysis_report.append(message)

    def _save_report_to_markdown(self):
        """将分析报告保存为 Markdown 文件"""
        with open('analysis_report.md', 'w', encoding='utf-8') as f:
            f.write("# 基金分析报告\n\n")
            f.write(f"**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for message in self.analysis_report:
                f.write(f"{message}\n\n")
        logger.info("分析报告已保存至 analysis_report.md")

    def analyze_multiple_funds(self, csv_url: str, personal_strategy: dict, code_column: str = 'code', max_funds: int = None):
        """批量分析 CSV 文件中的基金。"""
        try:
            funds_df = pd.read_csv(csv_url, encoding='gbk')
            self._log(f"导入CSV成功，共 {len(funds_df)} 个基金代码")
            
            # 确保列名存在
            required_columns = ['code', 'name', 'rose_3y', 'rank_r_3y', '类型']
            for col in required_columns:
                if col not in funds_df.columns:
                    self._log(f"错误：CSV文件缺少必需列 '{col}'")
                    return None
            
            funds_df['code'] = funds_df['code'].astype(str).str.zfill(6)
            fund_info_dict = funds_df.set_index('code').to_dict('index')
            fund_codes = funds_df['code'].unique().tolist()
            if max_funds:
                fund_codes = fund_codes[:max_funds]
                self._log(f"限制分析前 {max_funds} 个基金...")
        except Exception as e:
            self._log(f"导入 CSV 失败: {e}")
            return None

        market_data = self.data_fetcher.get_market_sentiment()
        strategy_engine = InvestmentStrategy(market_data, personal_strategy)

        results = []
        for code in fund_codes:
            fund_info = fund_info_dict.get(code, {})
            self._log(f"\n--- 正在分析基金 {code} ({fund_info.get('name', '未知')}) ---")
            
            fund_data = self.data_fetcher.get_real_time_fund_data(code, fund_info.get('name', '未知'))
            manager_data = self.data_fetcher.get_fund_manager_data(code)
            holdings_data = self.data_fetcher.get_fund_holdings_data(code)
            fund_metadata = self.data_fetcher.get_fund_info(code)
            if fund_info.get('类型') == '指数型':
                fund_metadata.update(self.data_fetcher.get_index_valuation(code))
            
            score, points_log = strategy_engine.score_fund(fund_data, fund_info, manager_data, holdings_data, fund_metadata)
            decision = strategy_engine.make_decision(score)
            
            results.append({
                'fund_code': code,
                'fund_name': fund_info.get('name', '未知'),
                'rose_3y': fund_info.get('rose_3y', np.nan),
                'rank_r_3y': fund_info.get('rank_r_3y', np.nan),
                'sharpe_ratio': fund_data.get('sharpe_ratio', np.nan),
                'max_drawdown': fund_data.get('max_drawdown', np.nan),
                'manager_name': manager_data.get('name', 'N/A'),
                'scale': fund_metadata.get('scale', np.nan),
                'years_since_establishment': fund_metadata.get('years_since_establishment', np.nan),
                'decision': decision,
                'score': score
            })
            self._log(f"评分详情: {points_log}")
            time.sleep(1)

        results_df = pd.DataFrame(results)
        
        self._log("\n--- 批量基金分析报告 ---")
        self._log(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"市场趋势: {market_data.get('trend', 'unknown')}")
        
        top_funds = results_df.sort_values('score', ascending=False).head(10)
        if not top_funds.empty:
            self._log("\n--- 推荐基金（综合评分前10名）---")
            self._log("\n" + top_funds[['decision', 'score', 'fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'sharpe_ratio', 'scale']].to_markdown(index=False))
        else:
            self._log("\n没有基金获得有效评分。")
        
        self._log("\n--- 所有基金分析结果 ---")
        self._log("\n" + results_df[['decision', 'score', 'fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'scale']].to_markdown(index=False))
        
        # 保存分析报告到 Markdown 文件
        self._save_report_to_markdown()
        
        return results_df

if __name__ == '__main__':
    # 请确保已安装 akshare, pandas, numpy, requests, beautifulsoup4 等库
    # pip install akshare pandas numpy requests beautifulsoup4
    
    # 请将 CSV 文件 URL 替换为您实际的基金列表文件
    CSV_URL = "https://raw.githubusercontent.com/qjlxg/own/main/recommended_cn_funds.csv"
    analyzer = FundAnalyzer(cache_data=True)
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }
    results_df = analyzer.analyze_multiple_funds(CSV_URL, my_personal_strategy, max_funds=2000)
