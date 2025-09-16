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
from collections import defaultdict
import logging
import asyncio
import aiohttp
import matplotlib.pyplot as plt
from scipy import stats  # 用于异常值检测

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('fund_analyzer.log'), logging.StreamHandler()]
)

class FundAnalyzer:
    """
    一个用于综合分析基金投资的工具类。
    能从 CSV 文件导入基金代码，结合收益率、排名、实时数据、基金经理、基金持仓和Beta系数进行投资决策。
    支持异步数据获取和可视化报告。
    """
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json', cache_expiry_days: int = 7):
        self.fund_data = {}
        self.fund_info = {}  # 存储 CSV 数据
        self.market_data = {}
        self.manager_data = {}
        self.holdings_data = {}  # 存储持仓数据
        self.beta_data = {}  # 新增：存储Beta系数
        self.analysis_report = []
        self.cache_data = cache_data
        self.cache_expiry_days = cache_expiry_days
        self.risk_free_rate = self._get_risk_free_rate()
        self.cache = self._load_cache(cache_file)
        self.cache_file = cache_file

    def _log(self, message: str, level: str = 'info'):
        """日志记录"""
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        self.analysis_report.append(message)

    def _get_risk_free_rate(self) -> float:
        """从东方财富获取最新 10 年期国债收益率作为无风险利率"""
        try:
            bond_data = ak.bond_zh_us_rate()
            risk_free_rate = bond_data[bond_data['item_name'] == '中国10年期国债']['value'].iloc[-1] / 100
            self._log(f"获取最新无风险利率：{risk_free_rate:.4f}")
            return risk_free_rate
        except Exception as e:
            self._log(f"获取无风险利率失败，使用默认值 0.018298: {e}", 'error')
            return 0.018298

    def _load_cache(self, cache_file: str) -> dict:
        """从文件加载缓存，并检查过期"""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                # 检查缓存时间
                if 'timestamp' in cache and datetime.now() - datetime.fromisoformat(cache['timestamp']) > timedelta(days=self.cache_expiry_days):
                    self._log("缓存已过期，重新创建。", 'warning')
                    return {}
                return cache.get('data', {})
            except (json.JSONDecodeError, ValueError) as e:
                self._log(f"缓存文件加载失败，将创建新缓存: {e}", 'error')
                return {}
        return {}

    def _save_cache(self):
        """将缓存保存到文件，添加时间戳"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': datetime.now().isoformat(), 'data': self.cache}, f, indent=4, ensure_ascii=False)

    async def _async_get(self, session, url, headers):
        """异步GET请求"""
        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            self._log(f"异步请求 {url} 失败: {e}", 'error')
            return None

    async def get_real_time_fund_data(self, fund_code: str):
        """获取单个基金的实时数据（净值、夏普比率、最大回撤），异步版本"""
        cache_key = f'fund_{fund_code}'
        if self.cache_data and cache_key in self.cache:
            self._log(f"使用缓存数据 for 基金 {fund_code}")
            self.fund_data[fund_code] = self.cache[cache_key]
            return True

        self._log(f"正在获取基金 {fund_code} 的实时数据...")
        for attempt in range(3):
            try:
                fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
                fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
                fund_data.set_index('净值日期', inplace=True)
                fund_data = fund_data.dropna()

                if len(fund_data) < 252:
                    raise ValueError("数据不足，无法计算可靠的夏普比率和回撤")

                returns = fund_data['单位净值'].pct_change().dropna()
                # 异常值检测：使用Z分数过滤
                z_scores = np.abs(stats.zscore(returns))
                returns = returns[z_scores < 3]  # 去除超过3倍标准差的值

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
                    self.cache[cache_key] = self.fund_data[fund_code]
                    self._save_cache()
                self._log(f"基金 {fund_code} 数据已获取：{self.fund_data[fund_code]}")
                return True
            except Exception as e:
                self._log(f"获取基金 {fund_code} 数据失败 (尝试 {attempt+1}/3): {e}", 'warning')
                await asyncio.sleep(2)
        self.fund_data[fund_code] = {'latest_nav': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
        return False

    async def _scrape_manager_data_from_web(self, fund_code: str) -> dict:
        """异步网页抓取基金经理数据"""
        self._log(f"尝试异步网页抓取基金 {fund_code} 的基金经理数据...")
        manager_url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        async with aiohttp.ClientSession() as session:
            html = await self._async_get(session, manager_url, headers)
            if not html:
                return None

            soup = BeautifulSoup(html, 'html.parser')
            title_label = soup.find('label', string='基金经理变动一览')
            if not title_label:
                self._log(f"在 {manager_url} 中未找到基金经理变动表格的标题。", 'warning')
                return None

            manager_table = title_label.find_parent().find_next_sibling('table')
            if not manager_table:
                self._log(f"在 {manager_url} 中未找到基金经理变动表格。", 'warning')
                return None

            rows = manager_table.find_all('tr')
            if len(rows) < 2:
                self._log("基金经理变动表格数据不完整。", 'warning')
                return None

            latest_manager_row = rows[1]
            cols = latest_manager_row.find_all('td')
            if len(cols) < 5:
                self._log("基金经理变动表格列数不正确。", 'warning')
                return None

            manager_name = cols[2].text.strip()
            tenure_str = cols[3].text.strip()
            cumulative_return_str = cols[4].text.strip()

            tenure_days = np.nan
            if '年又' in tenure_str:
                tenure_parts = tenure_str.split('年又')
                years = float(re.search(r'\d+', tenure_parts[0]).group())
                days = float(re.search(r'\d+', tenure_parts[1]).group())
                tenure_days = years * 365 + days
            elif '天' in tenure_str:
                tenure_days = float(re.search(r'\d+', tenure_str).group())
            elif '年' in tenure_str:
                tenure_days = float(re.search(r'\d+', tenure_str).group()) * 365

            cumulative_return = float(re.search(r'[-+]?\d*\.?\d+', cumulative_return_str).group()) if '%' in cumulative_return_str else np.nan

            return {
                'name': manager_name,
                'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                'cumulative_return': cumulative_return
            }

    async def get_fund_manager_data(self, fund_code: str):
        """获取基金经理数据（akshare优先，失败则异步抓取）"""
        self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
        try:
            manager_info = ak.fund_manager_em()  # 更新为fund_manager_em
            if not manager_info.empty:
                # 过滤特定基金
                manager_info = manager_info[manager_info['现任基金代码'].str.contains(fund_code, na=False)]
                if not manager_info.empty:
                    latest_manager = manager_info.sort_values(by='累计从业时间', ascending=False).iloc[0]  # 假设累计从业时间排序
                    name = latest_manager.get('姓名', 'N/A')
                    tenure_days = latest_manager.get('累计从业时间', np.nan)
                    cumulative_return = latest_manager.get('现任基金最佳回报', '0%')
                    cumulative_return = float(str(cumulative_return).replace('%', '')) if isinstance(cumulative_return, str) else float(cumulative_return)

                    self.manager_data[fund_code] = {
                        'name': name,
                        'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                        'cumulative_return': cumulative_return
                    }
                    self._log(f"基金 {fund_code} 经理数据已通过akshare获取：{self.manager_data[fund_code]}")
                    return True
        except Exception as e:
            self._log(f"使用akshare获取基金 {fund_code} 经理数据失败: {e}", 'warning')

        # 异步抓取
        scraped_data = await self._scrape_manager_data_from_web(fund_code)
        if scraped_data:
            self.manager_data[fund_code] = scraped_data
            self._log(f"基金 {fund_code} 经理数据已通过网页抓取获取：{self.manager_data[fund_code]}")
            return True
        else:
            self.manager_data[fund_code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
            return False

    async def get_market_sentiment(self):
        """异步获取市场情绪"""
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
            sentiment_score = price_change * 50 + volume_change * 50  # 新增：量化分数 (-100 to 100)
            if sentiment_score > 10:
                sentiment, trend = 'optimistic', 'bullish'
            elif sentiment_score < -10:
                sentiment, trend = 'pessimistic', 'bearish'
            else:
                sentiment, trend = 'neutral', 'neutral'

            self.market_data = {'sentiment': sentiment, 'trend': trend, 'score': sentiment_score}
            self._log(f"市场情绪数据已获取：{self.market_data}")
            return True
        except Exception as e:
            self._log(f"获取市场数据失败: {e}", 'error')
            self.market_data = {'sentiment': 'unknown', 'trend': 'unknown', 'score': 0}
            return False

    async def _scrape_holdings_data_from_web(self, fund_code: str) -> list:
        """异步网页抓取持仓数据"""
        self._log(f"尝试异步网页抓取基金 {fund_code} 的持仓数据...")
        holdings_url = f"http://fundf10.eastmoney.com/ccmx_{fund_code}.html"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        async with aiohttp.ClientSession() as session:
            html = await self._async_get(session, holdings_url, headers)
            if not html:
                return []

            soup = BeautifulSoup(html, 'html.parser')
            holdings_header = soup.find('h4', string=lambda t: t and '股票投资明细' in t)
            if not holdings_header:
                self._log("未找到持仓表格标题。", 'warning')
                return []

            holdings_table = holdings_header.find_next('table')
            if not holdings_table:
                self._log("未找到持仓表格。", 'warning')
                return []

            rows = holdings_table.find_all('tr')[1:]
            holdings = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    holdings.append({
                        '股票代码': cols[1].text.strip(),
                        '股票名称': cols[2].text.strip(),
                        '占净值比例': float(cols[4].text.strip().replace('%', '')),
                        '持仓市值（万元）': float(cols[6].text.strip().replace(',', '')),
                    })
            return holdings

    async def get_fund_holdings_data(self, fund_code: str):
        """获取基金持仓数据（akshare优先，失败则异步抓取）"""
        self._log(f"正在获取基金 {fund_code} 的持仓数据...")
        current_year = str(datetime.now().year)
        try:
            holdings_df = ak.fund_portfolio_hold_em(symbol=fund_code, date=current_year)  # 添加date参数
            if not holdings_df.empty:
                self.holdings_data[fund_code] = holdings_df.to_dict('records')
                self._log(f"基金 {fund_code} 持仓数据已通过akshare获取。")
                return True
        except Exception as e:
            self._log(f"通过akshare获取基金 {fund_code} 持仓数据失败: {e}", 'warning')

        scraped_holdings = await self._scrape_holdings_data_from_web(fund_code)
        if scraped_holdings:
            self.holdings_data[fund_code] = scraped_holdings
            self._log(f"基金 {fund_code} 持仓数据已通过网页抓取获取。")
            return True
        else:
            self.holdings_data[fund_code] = []
            return False

    async def get_fund_beta(self, fund_code: str):
        """计算基金Beta系数（相对于上证指数）"""
        try:
            fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            fund_returns = fund_data.set_index('净值日期')['单位净值'].pct_change().dropna()

            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            index_returns = index_data.set_index('date')['close'].pct_change().dropna()

            # 对齐日期
            combined = pd.concat([fund_returns, index_returns], axis=1, join='inner').dropna()
            cov = np.cov(combined.iloc[:,0], combined.iloc[:,1])[0][1]
            var = np.var(combined.iloc[:,1])
            beta = cov / var if var != 0 else np.nan
            self.beta_data[fund_code] = beta
            self._log(f"基金 {fund_code} Beta系数计算完成：{beta:.2f}")
            return True
        except Exception as e:
            self._log(f"计算基金 {fund_code} Beta失败: {e}", 'warning')
            self.beta_data[fund_code] = np.nan
            return False

    def make_decision(self, fund_code: str, personal_strategy: dict) -> str:
        """根据基金数据、CSV 数据、持仓、Beta和个人策略做出投资决策"""
        self._log(f"开始做出 {fund_code} 的投资决策:")
        if fund_code not in self.fund_data or not self.market_data:
            return "数据获取不完整，无法给出明确建议。"

        market_trend = self.market_data.get('trend', 'unknown')
        market_score = self.market_data.get('score', 0)
        fund_drawdown = self.fund_data[fund_code].get('max_drawdown', float('inf'))
        invest_horizon = personal_strategy.get('horizon', 'unknown')
        risk_tolerance = personal_strategy.get('risk_tolerance', 'medium')
        sharpe_ratio = self.fund_data[fund_code].get('sharpe_ratio', 0)
        beta = self.beta_data.get(fund_code, np.nan)

        fund_info = self.fund_info.get(fund_code, {})
        rose_5y = fund_info.get('rose(5y)', np.nan)
        rank_r_5y = fund_info.get('rank_r(5y)', np.nan)
        fund_name = fund_info.get('名称', '未知')

        manager_trust = False
        manager_return = self.manager_data.get(fund_code, {}).get('cumulative_return', np.nan)
        tenure_years = self.manager_data.get(fund_code, {}).get('tenure_years', np.nan)
        if pd.notna(manager_return) and pd.notna(tenure_years):
            if tenure_years > 5 or manager_return > 20:
                manager_trust = True
            self._log(f"基金经理 {self.manager_data[fund_code]['name']} 任职 {tenure_years:.2f} 年，累计回报 {manager_return:.2f}%，信任度：{manager_trust}")

        holdings = self.holdings_data.get(fund_code, [])
        holdings_report = ""
        fund_risk_high = False
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            top_10_holdings_sum = holdings_df['占净值比例'].iloc[:10].sum()
            holdings_report += f"前十持仓集中度为 {top_10_holdings_sum:.2f}%。"
            if top_10_holdings_sum > 50:  # 调整阈值
                holdings_report += "集中度较高，风险偏大。"
                fund_risk_high = True
            else:
                holdings_report += "集中度适中，风险可控。"

            # 新增：尝试获取行业分配
            try:
                industry_df = ak.fund_portfolio_industry_allocation_em(symbol=fund_code, date=str(datetime.now().year))
                if not industry_df.empty:
                    top_industry_concentration = industry_df['占净值比例'].iloc[:3].sum()
                    if top_industry_concentration > 70:
                        holdings_report += f" 前三行业集中度 {top_industry_concentration:.2f}%，行业风险高。"
                        fund_risk_high = True
            except Exception:
                pass

            self._log(f"持仓分析：{holdings_report}")

        # 决策逻辑（丰富版）
        is_high_beta = pd.notna(beta) and beta > 1.5
        if invest_horizon == 'long-term':
            if market_trend == 'bearish' or market_score < -10:
                if fund_drawdown <= 0.2 and not is_high_beta and (risk_tolerance in ['medium', 'high'] or manager_trust):
                    return f"适合长期布局：市场熊市（分数 {market_score:.2f}），{fund_name} 回撤 {fund_drawdown:.2f} 控制良好，Beta {beta:.2f}。建议分批买入。"
                else:
                    return f"观望：市场熊市，{fund_name} 回撤 {fund_drawdown:.2f}，Beta {beta:.2f} 高风险。建议选择更稳健的基金。"
            else:
                is_top_performer = (sharpe_ratio > 1.0 or pd.notna(rose_5y) and rose_5y > 50 or manager_trust)
                if is_top_performer and pd.notna(rank_r_5y) and rank_r_5y < 0.05 and risk_tolerance != 'low':
                    if fund_risk_high or is_high_beta:
                        return f"谨慎加仓：市场 {market_trend}（分数 {market_score:.2f}），{fund_name} 表现优异，但持仓/行业集中或Beta {beta:.2f} 高。建议谨慎加仓。"
                    return f"继续持有或加仓：市场 {market_trend}，{fund_name} 表现优异，5年回报 {rose_5y:.2f}% （排名前 {rank_r_5y*100:.2f}%），Beta {beta:.2f}。"
                else:
                    return f"评估其他基金：市场 {market_trend}，但 {fund_name} 表现平平（夏普 {sharpe_ratio:.2f}，5年排名 {rank_r_5y*100:.2f}%）。"
        elif invest_horizon == 'short-term':
            if sharpe_ratio > 1.5 and market_trend == 'bullish' and market_score > 10 and risk_tolerance != 'low' and not is_high_beta:
                if fund_risk_high:
                    return f"适量买入但注意风险：市场牛市，{fund_name} 夏普 {sharpe_ratio:.2f}，Beta {beta:.2f}，但持仓集中高。"
                return f"适量买入：市场牛市，{fund_name} 夏普 {sharpe_ratio:.2f}，Beta {beta:.2f} 适合短期。"
            else:
                return f"保持谨慎：市场 {market_trend} 或 {fund_name} 不适合短期投资（Beta {beta:.2f}）。"
        return "重新审视策略：投资策略与市场状况不匹配。"

    def generate_visual_report(self, fund_code: str, output_dir: str = 'visual_reports'):
        """生成可视化报告：净值走势和持仓饼图"""
        os.makedirs(output_dir, exist_ok=True)

        # 净值走势图
        try:
            fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            plt.figure(figsize=(10, 5))
            plt.plot(fund_data['净值日期'], fund_data['单位净值'], label='单位净值')
            plt.title(f"{fund_code} 净值走势")
            plt.xlabel('日期')
            plt.ylabel('单位净值')
            plt.legend()
            plt.savefig(f"{output_dir}/{fund_code}_nav.png")
            plt.close()
            self._log(f"生成 {fund_code} 净值走势图。")
        except Exception as e:
            self._log(f"生成净值图失败: {e}", 'warning')

        # 持仓饼图
        holdings = self.holdings_data.get(fund_code, [])
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            plt.figure(figsize=(8, 8))
            plt.pie(holdings_df['占净值比例'][:10], labels=holdings_df['股票名称'][:10], autopct='%1.1f%%')
            plt.title(f"{fund_code} 前十持仓分布")
            plt.savefig(f"{output_dir}/{fund_code}_holdings_pie.png")
            plt.close()
            self._log(f"生成 {fund_code} 持仓饼图。")

    async def analyze_multiple_funds(self, csv_url: str, personal_strategy: dict, code_column: str = '代码', max_funds: int = 100):
        """批量异步分析基金"""
        self._log("正在从 CSV 导入基金代码列表...")
        try:
            funds_df = pd.read_csv(csv_url, encoding='gbk')
            self._log(f"导入成功，共 {len(funds_df)} 个基金代码")

            funds_df[code_column] = funds_df[code_column].astype(str).str.zfill(6)
            self.fund_info = funds_df.set_index(code_column).to_dict('index')

            fund_codes = funds_df[code_column].unique().tolist()[:max_funds]
            self._log(f"分析前 {len(fund_codes)} 个基金：{fund_codes}...")
        except Exception as e:
            self._log(f"导入 CSV 失败: {e}", 'error')
            return None

        await self.get_market_sentiment()

        tasks = []
        for code in fund_codes:
            tasks.append(self.get_real_time_fund_data(code))
            tasks.append(self.get_fund_manager_data(code))
            tasks.append(self.get_fund_holdings_data(code))
            tasks.append(self.get_fund_beta(code))

        await asyncio.gather(*tasks)

        results = []
        for code in fund_codes:
            decision = self.make_decision(code, personal_strategy)
            fund_info = self.fund_info.get(code, {})
            results.append({
                'fund_code': code,
                'fund_name': fund_info.get('名称', '未知'),
                'rose_5y': fund_info.get('rose(5y)', np.nan),
                'rank_r_5y': fund_info.get('rank_r(5y)', np.nan),
                'latest_nav': self.fund_data.get(code, {}).get('latest_nav', np.nan),
                'sharpe_ratio': self.fund_data.get(code, {}).get('sharpe_ratio', np.nan),
                'max_drawdown': self.fund_data.get(code, {}).get('max_drawdown', np.nan),
                'beta': self.beta_data.get(code, np.nan),
                'manager_name': self.manager_data.get(code, {}).get('name', 'N/A'),
                'manager_return': self.manager_data.get(code, {}).get('cumulative_return', np.nan),
                'tenure_years': self.manager_data.get(code, {}).get('tenure_years', np.nan),
                'market_trend': self.market_data.get('trend', 'unknown'),
                'decision': decision,
                'top_10_holdings': self.holdings_data.get(code, [])[:10]
            })
            self.generate_visual_report(code)  # 生成可视化

        results_df = pd.DataFrame(results)

        print("\n--- 批量基金分析报告 ---")
        print(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"市场趋势: {self.market_data.get('trend', 'unknown')} (分数: {self.market_data.get('score', 0):.2f})")

        top_funds = results_df[results_df['rank_r_5y'] < 0.01].sort_values('rank_r_5y')
        if not top_funds.empty:
            print("\n--- 推荐基金（5年排名前 1%）---")
            print(top_funds[['decision', 'fund_code', 'fund_name', 'rose_5y', 'rank_r_5y', 'sharpe_ratio', 'max_drawdown', 'beta', 'manager_name', 'manager_return', 'tenure_years']].to_string(index=False))
        else:
            print("\n没有基金满足 5 年排名前 1% 的条件。")

        print("\n--- 所有基金分析结果 ---")
        print(results_df[['decision', 'fund_code', 'fund_name', 'rose_5y', 'rank_r_5y', 'sharpe_ratio', 'max_drawdown', 'beta', 'manager_name']].to_string(index=False))

        print("-" * 25)

        for result in results:
            if result.get('top_10_holdings'):
                print(f"\n--- 基金 {result['fund_code']} ({result['fund_name']}) 前十持仓 ---")
                holdings_df = pd.DataFrame(result['top_10_holdings'])
                print(holdings_df.to_string(index=False))
        print("-" * 25)

        print("可视化报告已保存到 'visual_reports' 目录。")

        return results_df

if __name__ == '__main__':
    CSV_URL = "https://raw.githubusercontent.com/qjlxg/rep/main/recommended_cn_funds.csv"
    analyzer = FundAnalyzer(cache_data=True)
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }
    # 运行异步分析
    asyncio.run(analyzer.analyze_multiple_funds(CSV_URL, my_personal_strategy, code_column='代码', max_funds=88))
