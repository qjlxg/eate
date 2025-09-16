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

class FundDataFetcher:
    """负责所有数据获取、清洗和缓存管理。"""
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json'):
        self.cache_data = cache_data
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.risk_free_rate = self._get_risk_free_rate()
        self._web_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    def _log(self, message: str):
        print(f"[数据获取] {message}")

    def _load_cache(self) -> dict:
        """从文件加载缓存并检查时效性。"""
        if not os.path.exists(self.cache_file):
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
            
    def _scrape_fund_info(self, fund_code: str) -> dict:
        """从天天基金网抓取基金规模、评级和跟踪误差等信息。"""
        info_url = f"http://fund.eastmoney.com/{fund_code}.html"
        info = {}
        try:
            response = requests.get(info_url, headers=self._web_headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 抓取基金规模
            size_text_elem = soup.find('div', class_='dataItem02')
            if size_text_elem:
                size_text = size_text_elem.find('dd').get_text(strip=True)
                size_value = re.search(r'规模：(\d+\.\d+)亿元', size_text)
                if size_value:
                    info['size_billion'] = float(size_value.group(1))

            # 抓取基金评级（以晨星评级为例）
            rating_elements = soup.find_all('div', class_='rating-star')
            if rating_elements:
                rating_class = rating_elements[0].find('span')['class'][0]
                info['rating'] = int(re.search(r'\d+', rating_class).group())

            # 抓取跟踪误差（仅限指数基金）
            try:
                tracking_error_elem = soup.find(string=re.compile("跟踪误差")).find_next('span')
                if tracking_error_elem:
                    tracking_error_text = tracking_error_elem.get_text()
                    info['tracking_error'] = float(tracking_error_text.replace('%', ''))
            except:
                info['tracking_error'] = np.nan
        
        except Exception as e:
            self._log(f"网页抓取基金 {fund_code} 附加信息失败: {e}")

        return info

    def get_real_time_fund_data(self, fund_code: str, fund_type: str = '股票型') -> dict:
        """获取单个基金的实时数据和性能指标。"""
        if self.cache_data and fund_code in self.cache.get('fund', {}):
            return self.cache['fund'][fund_code]

        self._log(f"正在获取基金 {fund_code} 的实时数据和性能指标...")
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
                
                metrics = self._scrape_performance_metrics(fund_code)
                info = self._scrape_fund_info(fund_code)
                
                data = {
                    'latest_nav': float(fund_data_ak['单位净值'].iloc[-1]),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'annual_return': metrics.get('annual_return', np.nan),
                    'volatility': metrics.get('volatility', np.nan),
                    'alpha': metrics.get('alpha', np.nan),
                    'beta': metrics.get('beta', np.nan),
                    'size_billion': info.get('size_billion', np.nan),
                    'rating': info.get('rating', np.nan),
                    'tracking_error': info.get('tracking_error', np.nan)
                }
                
                if self.cache_data:
                    self.cache.setdefault('fund', {})[fund_code] = data
                    self._save_cache()
                return data
            except Exception as e:
                self._log(f"获取基金 {fund_code} 数据失败 (尝试 {attempt+1}/3): {e}")
                time.sleep(2)
        return {}

    def _scrape_performance_metrics(self, fund_code: str) -> dict:
        """从天天基金网抓取年化收益率、波动率、Alpha和Beta等数据"""
        metrics = {'annual_return': np.nan, 'volatility': np.nan, 'alpha': np.nan, 'beta': np.nan}
        metrics_url = f"http://fundf10.eastmoney.com/jdzc_{fund_code}.html"
        try:
            response = requests.get(metrics_url, headers=self._web_headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'fxtb'})
            if not table: return metrics
            
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    metric_name = cols[0].text.strip()
                    metric_value_str = cols[1].text.strip()
                    
                    if '年化收益' in metric_name and '%' in metric_value_str:
                        metrics['annual_return'] = float(metric_value_str.replace('%', ''))
                    elif '年化波动率' in metric_name and '%' in metric_value_str:
                        metrics['volatility'] = float(metric_value_str.replace('%', ''))
                    elif '阿尔法' in metric_name:
                        metrics['alpha'] = float(metric_value_str)
                    elif '贝塔' in metric_name:
                        metrics['beta'] = float(metric_value_str)
            return metrics
        except Exception as e:
            self._log(f"抓取基金 {fund_code} 性能指标失败: {e}")
            return metrics

    def get_fund_manager_data(self, fund_code: str) -> dict:
        """获取基金经理数据。"""
        if self.cache_data and fund_code in self.cache.get('manager', {}):
            return self.cache['manager'][fund_code]
        
        self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
        manager_data = self._scrape_manager_data_from_web(fund_code)
        if self.cache_data:
            self.cache.setdefault('manager', {})[fund_code] = manager_data
            self._save_cache()
        return manager_data

    def _scrape_manager_data_from_web(self, fund_code: str) -> dict:
        """从天天基金网通过网页抓取获取基金经理数据。"""
        manager_url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        try:
            response = requests.get(manager_url, headers=self._web_headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('label', string='基金经理变动一览').find_parent().find_next_sibling('table')
            if not table or len(table.find_all('tr')) < 2: return {}
            
            latest_manager_row = table.find_all('tr')[1]
            cols = latest_manager_row.find_all('td')
            if len(cols) < 5: return {}

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
            return data
        except Exception as e:
            self._log(f"获取基金 {fund_code} 持仓数据失败: {e}")
            return {'holdings': [], 'sectors': []}

class InvestmentStrategy:
    """封装所有投资决策逻辑。"""
    def __init__(self, market_data: dict, personal_strategy: dict):
        self.market_data = market_data
        self.personal_strategy = personal_strategy

    def _log(self, message: str):
        print(f"[决策引擎] {message}")

    def score_fund(self, fund_data: dict, fund_info: dict, manager_data: dict, holdings_data: dict) -> float:
        """
        基于多维指标为基金打分，整合用户笔记中的筛选标准。
        分数越高，推荐度越高。
        """
        score = 0
        market_trend = self.market_data.get('trend', 'neutral')
        fund_type = fund_info.get('类型', '未知')

        # 1. 基金通用评分（适用于所有类型）
        sharpe_ratio = fund_data.get('sharpe_ratio')
        annual_return = fund_data.get('annual_return')
        max_drawdown = fund_data.get('max_drawdown')
        rating = fund_data.get('rating')

        self._log(f"  - 基金通用指标: 夏普比率: {sharpe_ratio}, 年化收益: {annual_return}, 最大回撤: {max_drawdown}, 评级: {rating}")
        
        # 收益与风险评分
        if pd.notna(annual_return) and annual_return > 15: score += 20; self._log("    - 得分: 年化收益 > 15%")
        elif pd.notna(annual_return) and annual_return > 10: score += 10; self._log("    - 得分: 年化收益 > 10%")
        if pd.notna(sharpe_ratio) and sharpe_ratio > 1.5: score += 20; self._log("    - 得分: 夏普比率 > 1.5")
        elif pd.notna(sharpe_ratio) and sharpe_ratio > 1.0: score += 10; self._log("    - 得分: 夏普比率 > 1.0")
        if pd.notna(max_drawdown) and max_drawdown < 0.2: score += 15; self._log("    - 得分: 最大回撤 < 20%")

        # 评级评分
        if pd.notna(rating) and rating >= 4: score += rating * 5; self._log(f"    - 得分: 评级为 {rating} 星")
        
        # 2. 针对特定基金类型的评分
        fund_size = fund_data.get('size_billion')
        if '股票' in fund_type or '混合' in fund_type:
            self._log(f"  - 股票/混合基金指标: 规模: {fund_size}亿")
            if pd.notna(fund_size) and fund_size > 10: score += 10; self._log("    - 得分: 规模 > 10亿")

        if '债券' in fund_type:
            tenure_years = manager_data.get('tenure_years')
            self._log(f"  - 债券基金指标: 基金经理任职年限: {tenure_years}年, 评级: {rating}")
            if pd.notna(tenure_years) and tenure_years < 3:
                score -= 20; self._log("    - 扣分: 基金经理任职 < 3年")
            if pd.isna(rating): score -= 10; self._log("    - 扣分: 暂无评级")

        if '指数' in fund_type:
            tracking_error = fund_data.get('tracking_error')
            self._log(f"  - 指数基金指标: 规模: {fund_size}亿, 跟踪误差: {tracking_error}%")
            if pd.notna(tracking_error):
                if tracking_error < 0.5: score += 20; self._log("    - 得分: 跟踪误差 < 0.5%")
                elif tracking_error < 1.0: score += 10; self._log("    - 得分: 跟踪误差 < 1.0%")
            if pd.notna(fund_size) and fund_size < 1: score -= 20; self._log("    - 扣分: 规模 < 1亿")
        
        # 3. 基金经理评分
        manager_name = manager_data.get('name', 'N/A')
        tenure_years = manager_data.get('tenure_years')
        manager_return = manager_data.get('cumulative_return')
        self._log(f"  - 基金经理指标: 姓名: {manager_name}, 任职年限: {tenure_years}年, 任职回报: {manager_return}%")
        if pd.notna(tenure_years) and tenure_years > 3 and pd.notna(manager_return) and manager_return > 20:
            score += 20; self._log("    - 得分: 基金经理资深且回报高")
        
        # 4. 持仓评分（反向指标，风险越高扣分）
        holdings = holdings_data.get('holdings', [])
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            top_10_concentration = holdings_df['占净值比例'].iloc[:10].sum() if len(holdings_df) >= 10 else holdings_df['占净值比例'].sum()
            if top_10_concentration > 60:
                score -= 15; self._log(f"  - 扣分: 前十持仓集中度 > 60% ({top_10_concentration:.2f}%)")
        
        # 5. 市场趋势调整
        original_score = score
        if market_trend == 'bullish':
            score *= 1.1 # 牛市中看重进攻性，分数略上调
            self._log(f"  - 市场趋势调整: 牛市，分数上调至 {score:.2f}")
        elif market_trend == 'bearish':
            score *= 0.9 # 熊市中看重防守性，分数略下调
            self._log(f"  - 市场趋势调整: 熊市，分数下调至 {score:.2f}")
        
        return score

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
        print(f"[分析报告] {message}")
        self.analysis_report.append(message)

    def analyze_multiple_funds(self, csv_url: str, personal_strategy: dict, code_column: str = '代码', max_funds: int = None):
        """批量分析 CSV 文件中的基金。"""
        try:
            funds_df = pd.read_csv(csv_url, encoding='gbk')
            self._log(f"导入成功，共 {len(funds_df)} 个基金代码")
            
            # 确保列名存在，以便兼容用户提供的CSV
            if 'rose(3y)' not in funds_df.columns:
                self._log("未找到 'rose(3y)' 列，将使用 'rose(5y)' 作为替代。")
                funds_df['rose(3y)'] = funds_df.get('rose(5y)', np.nan)
            if 'rank_r(3y)' not in funds_df.columns:
                self._log("未找到 'rank_r(3y)' 列，将使用 'rank_r(5y)' 作为替代。")
                funds_df['rank_r(3y)'] = funds_df.get('rank_r(5y)', np.nan)
            if '类型' not in funds_df.columns:
                self._log("警告: 未找到 '类型' 列。将无法进行特定类型的评分。")
                funds_df['类型'] = '未知'
            
            funds_df[code_column] = funds_df[code_column].astype(str).str.zfill(6)
            fund_info_dict = funds_df.set_index(code_column).to_dict('index')
            fund_codes = funds_df[code_column].unique().tolist()
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
            self._log(f"\n--- 正在分析基金 {code} ({fund_info.get('名称', '未知')}) ---")
            
            fund_data = self.data_fetcher.get_real_time_fund_data(code)
            manager_data = self.data_fetcher.get_fund_manager_data(code)
            holdings_data = self.data_fetcher.get_fund_holdings_data(code)
            
            score = strategy_engine.score_fund(fund_data, fund_info, manager_data, holdings_data)
            decision = strategy_engine.make_decision(score)
            
            results.append({
                'fund_code': code,
                'fund_name': fund_info.get('名称', '未知'),
                'rose_3y': fund_info.get('rose(3y)', np.nan),
                'rank_r_3y': fund_info.get('rank_r(3y)', np.nan),
                'sharpe_ratio': fund_data.get('sharpe_ratio', np.nan),
                'max_drawdown': fund_data.get('max_drawdown', np.nan),
                'manager_name': manager_data.get('name', 'N/A'),
                'decision': decision,
                'score': score
            })
            time.sleep(1)

        results_df = pd.DataFrame(results)
        
        self._log("\n--- 批量基金分析报告 ---")
        self._log(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"市场趋势: {market_data.get('trend', 'unknown')}")
        
        top_funds = results_df.sort_values('score', ascending=False).head(10)
        if not top_funds.empty:
            self._log("\n--- 推荐基金（综合评分前10名）---")
            print(top_funds[['decision', 'score', 'fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'sharpe_ratio']].to_string(index=False))
        else:
            self._log("\n没有基金获得有效评分。")
        
        self._log("\n--- 所有基金分析结果 ---")
        print(results_df[['decision', 'score', 'fund_code', 'fund_name', 'rose_3y', 'rank_r_3y']].to_string(index=False))
        
        return results_df

if __name__ == '__main__':
    # 请确保您已安装 akshare, pandas, numpy, requests, bs4 等库
    # pip install akshare pandas numpy requests beautifulsoup4
    
    # 请将 CSV 文件 URL 替换为您实际的基金列表文件
    CSV_URL = "https://raw.githubusercontent.com/qjlxg/rep/main/recommended_cn_funds.csv"
    analyzer = FundAnalyzer(cache_data=True)
    my_personal_strategy = {
        'horizon': 'long-term',
        'risk_tolerance': 'medium'
    }
    results_df = analyzer.analyze_multiple_funds(CSV_URL, my_personal_strategy, max_funds=10)
