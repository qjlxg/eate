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

class FundAnalyzer:
    """
    一个用于综合分析基金投资的工具类。
    能从 CSV 文件导入基金代码，结合收益率、排名、实时数据、基金经理和基金持仓数据进行投资决策。
    """
    def __init__(self, cache_data: bool = True, cache_file: str = 'fund_cache.json'):
        self.fund_data = {}
        self.fund_info = {}  # 存储 CSV 数据
        self.market_data = {}
        self.manager_data = {}
        self.holdings_data = {} # 新增：存储持仓数据
        self.analysis_report = []
        self.cache_data = cache_data
        self.risk_free_rate = self._get_risk_free_rate()
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _log(self, message: str):
        """将日志信息添加到报告列表中"""
        print(message)
        self.analysis_report.append(message)

    def _get_risk_free_rate(self) -> float:
        """从东方财富获取最新 10 年期国债收益率作为无风险利率"""
        try:
            # 修复 akshare 数据结构变化导致的问题
            bond_data = ak.bond_zh_us_rate()
            # 找到中国10年期国债的数据行，并获取最新值
            # 注意：akshare接口返回的列名经常变化，这里使用最新发现的列名'value'
            risk_free_rate = bond_data[bond_data['item_name'] == '中国10年期国债']['value'].iloc[-1] / 100
            self._log(f"获取最新无风险利率：{risk_free_rate:.4f}")
            return risk_free_rate
        except Exception as e:
            self._log(f"获取无风险利率失败，使用默认值 0.03: {e}")
            return 0.03

    def _load_cache(self) -> dict:
        """从文件加载缓存，并检查时间戳是否过期（按天过期）"""
        if os.path.exists(self.cache_file):
            try:
                # 获取文件的最后修改时间戳
                mod_time = datetime.fromtimestamp(os.path.getmtime(self.cache_file))
                # 检查是否超过一天
                if datetime.now() - mod_time > timedelta(days=1):
                    self._log(f"缓存文件 {self.cache_file} 已过期，将创建新缓存。")
                    return {}
                
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self._log(f"成功加载缓存文件 {self.cache_file}")
                    return cache_data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._log(f"缓存文件加载失败，可能是格式错误或编码问题。将创建新缓存。")
                return {}
        return {}

    def _save_cache(self):
        """将缓存保存到文件"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)
            self._log("缓存已保存。")

    def get_real_time_fund_data(self, fund_code: str):
        """获取单个基金的实时数据（净值、夏普比率、最大回撤）"""
        cache_key = f"fund_{fund_code}"
        if self.cache_data and cache_key in self.cache:
            self._log(f"使用缓存数据 for 基金 {fund_code}")
            self.fund_data[fund_code] = self.cache[cache_key]
            return True

        self._log(f"正在获取基金 {fund_code} 的实时数据...")
        for attempt in range(3):  # 手动重试机制，最多3次
            try:
                fund_data = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
                fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
                fund_data.set_index('净值日期', inplace=True)
                
                # 数据清洗：去除异常值和缺失值
                fund_data = fund_data.dropna()
                if len(fund_data) < 252:  # 至少一年数据
                    raise ValueError("数据不足，无法计算可靠的夏普比率和回撤")

                returns = fund_data['单位净值'].pct_change().dropna()
                
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
                self._log(f"获取基金 {fund_code} 数据失败 (尝试 {attempt+1}/3): {e}")
                time.sleep(2)  # 等待2秒后重试
        self.fund_data[fund_code] = {'latest_nav': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}
        return False

    def _scrape_manager_data_from_web(self, fund_code: str) -> dict:
        """
        从天天基金网通过网页抓取获取基金经理数据
        """
        self._log(f"尝试通过网页抓取获取基金 {fund_code} 的基金经理数据...")
        manager_url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(manager_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 找到包含“基金经理变动一览”文本的标签
            title_label = soup.find('label', string='基金经理变动一览')
            if not title_label:
                self._log(f"在 {manager_url} 中未找到基金经理变动表格的标题。")
                return None
            
            # 从父容器中找到表格
            manager_table = title_label.find_parent().find_next_sibling('table')
            if not manager_table:
                self._log(f"在 {manager_url} 中未找到基金经理变动表格。")
                return None
            
            rows = manager_table.find_all('tr')
            if len(rows) < 2:
                self._log("基金经理变动表格数据不完整。")
                return None
            
            # 找到第一行数据，即最新任职的经理
            latest_manager_row = rows[1]
            cols = latest_manager_row.find_all('td')
            
            if len(cols) < 5:
                self._log("基金经理变动表格列数不正确。")
                return None
            
            manager_name = cols[2].text.strip()
            tenure_str = cols[3].text.strip()
            cumulative_return_str = cols[4].text.strip()
            
            # 解析任职天数和累计回报
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
            else:
                tenure_days = np.nan
                
            cumulative_return = float(re.search(r'[-+]?\d*\.?\d+', cumulative_return_str).group()) if '%' in cumulative_return_str else np.nan

            return {
                'name': manager_name,
                'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                'cumulative_return': cumulative_return
            }
        except requests.exceptions.RequestException as e:
            self._log(f"网页抓取基金 {fund_code} 经理数据失败: {e}")
            return None
        except Exception as e:
            self._log(f"解析网页内容失败: {e}")
            return None

    def get_fund_manager_data(self, fund_code: str):
        """
        获取基金经理数据（首先尝试使用 akshare，失败则通过网页抓取）
        """
        cache_key = f"manager_{fund_code}"
        if self.cache_data and cache_key in self.cache:
            self._log(f"使用缓存数据 for 基金经理 {fund_code}")
            self.manager_data[fund_code] = self.cache[cache_key]
            return True

        self._log(f"正在获取基金 {fund_code} 的基金经理数据...")
        try:
            # 修复：akshare接口已变更为fund_manager_info_em
            manager_info = ak.fund_manager_info_em(fund_code=fund_code)
            if not manager_info.empty:
                latest_manager = manager_info.sort_values(by='上任日期', ascending=False).iloc[0]
                name = latest_manager.get('姓名', 'N/A')
                tenure_days = latest_manager.get('任职天数', np.nan)
                cumulative_return = latest_manager.get('任职回报', '0%')
                cumulative_return = float(str(cumulative_return).replace('%', '')) if isinstance(cumulative_return, str) else float(cumulative_return)
                
                self.manager_data[fund_code] = {
                    'name': name,
                    'tenure_years': float(tenure_days) / 365.0 if pd.notna(tenure_days) else np.nan,
                    'cumulative_return': cumulative_return
                }
                if self.cache_data:
                    self.cache[cache_key] = self.manager_data[fund_code]
                    self._save_cache()
                self._log(f"基金 {fund_code} 经理数据已通过akshare获取：{self.manager_data[fund_code]}")
                return True
        except Exception as e:
            self._log(f"使用akshare获取基金 {fund_code} 经理数据失败: {e}")
        
        # 如果akshare失败，尝试网页抓取
        scraped_data = self._scrape_manager_data_from_web(fund_code)
        if scraped_data:
            self.manager_data[fund_code] = scraped_data
            if self.cache_data:
                self.cache[cache_key] = self.manager_data[fund_code]
                self._save_cache()
            self._log(f"基金 {fund_code} 经理数据已通过网页抓取获取：{self.manager_data[fund_code]}")
            return True
        else:
            self.manager_data[fund_code] = {'name': 'N/A', 'tenure_years': np.nan, 'cumulative_return': np.nan}
            return False

    def get_market_sentiment(self):
        """获取市场情绪（仅调用一次，基于上证指数）"""
        cache_key = "market_sentiment"
        if self.cache_data and cache_key in self.cache:
            self.market_data = self.cache[cache_key]
            self._log("使用缓存的市场情绪数据")
            return True

        self._log("正在获取市场情绪数据...")
        try:
            index_data = ak.stock_zh_index_daily_em(symbol="sh000001")
            index_data['date'] = pd.to_datetime(index_data['date'])
            last_week_data = index_data.tail(5)
            
            # 计算过去一周的涨跌幅
            last_week_change = (last_week_data['close'].iloc[-1] - last_week_data['open'].iloc[0]) / last_week_data['open'].iloc[0]
            
            # 简单判断市场趋势
            trend = "上升" if last_week_change > 0.01 else "下降" if last_week_change < -0.01 else "盘整"
            
            self.market_data = {
                'latest_close': float(index_data['close'].iloc[-1]),
                'last_week_change': float(last_week_change),
                'trend': trend
            }
            if self.cache_data:
                self.cache[cache_key] = self.market_data
                self._save_cache()
            self._log(f"市场情绪数据已获取：{self.market_data}")
            return True
        except Exception as e:
            self._log(f"获取市场情绪数据失败: {e}")
            self.market_data = {'latest_close': np.nan, 'last_week_change': np.nan, 'trend': '未知'}
            return False

    def get_fund_holdings_data(self, fund_code: str):
        """
        新增：获取基金前十大持仓及行业分布数据
        """
        cache_key = f"holdings_{fund_code}"
        if self.cache_data and cache_key in self.cache:
            self._log(f"使用缓存数据 for 持仓 {fund_code}")
            self.holdings_data[fund_code] = self.cache[cache_key]
            return True

        self._log(f"正在获取基金 {fund_code} 的持仓数据...")
        
        try:
            # 基金持仓数据的 akshare 接口，包含行业分布
            holdings_data = ak.fund_main_hold_industry_distribution_em(symbol=fund_code, indicator="十大持仓")
            if holdings_data.empty:
                self._log(f"未找到基金 {fund_code} 的持仓数据。")
                self.holdings_data[fund_code] = {'top_10_holdings': [], 'industry_distribution': {}}
                return False

            # 处理持仓股票信息
            top_10_holdings = []
            if '股票名称' in holdings_data.columns and '占净值比例' in holdings_data.columns:
                holdings_df = holdings_data[['股票名称', '占净值比例']].rename(columns={'股票名称': 'stock_name', '占净值比例': 'proportion'})
                top_10_holdings = holdings_df.to_dict('records')

            # 处理行业分布信息
            industry_distribution = {}
            if '行业名称' in holdings_data.columns and '市值占净值比例' in holdings_data.columns:
                # 重新获取行业分布数据，因为十大持仓的接口可能不完整
                industry_data = ak.fund_main_hold_industry_distribution_em(symbol=fund_code, indicator="行业配置")
                if not industry_data.empty:
                    industry_data['占净值比例'] = industry_data['占净值比例'].astype(float)
                    industry_distribution = industry_data.set_index('行业类别')['占净值比例'].to_dict()

            self.holdings_data[fund_code] = {
                'top_10_holdings': top_10_holdings,
                'industry_distribution': industry_distribution
            }
            
            if self.cache_data:
                self.cache[cache_key] = self.holdings_data[fund_code]
                self._save_cache()
            self._log(f"基金 {fund_code} 持仓数据已获取。")
            return True
        except Exception as e:
            self._log(f"获取基金 {fund_code} 持仓数据失败: {e}")
            self.holdings_data[fund_code] = {'top_10_holdings': [], 'industry_distribution': {}}
            return False

    def load_funds_from_csv(self, file_path: str):
        """从 CSV 文件加载基金代码和名称"""
        try:
            self.fund_info = pd.read_csv(file_path, encoding='utf-8')
            self._log(f"成功从 {file_path} 加载基金信息。共 {len(self.fund_info)} 只基金。")
        except FileNotFoundError:
            self._log(f"错误: 文件 {file_path} 未找到。请检查路径。")
            self.fund_info = pd.DataFrame()
        except Exception as e:
            self._log(f"读取文件时发生错误: {e}")
            self.fund_info = pd.DataFrame()
            
    def analyze_funds(self):
        """执行所有基金的综合分析"""
        if self.fund_info.empty:
            self._log("没有基金信息可供分析。请先加载基金列表。")
            return
            
        self.get_market_sentiment()

        # 使用 for 循环逐个处理基金
        for index, row in self.fund_info.iterrows():
            fund_code = str(row['fund_code']).zfill(6)
            fund_name = row['fund_name']
            self._log("-" * 25)
            self._log(f"开始分析基金: {fund_name} ({fund_code})")
            
            # 获取实时数据
            self.get_real_time_fund_data(fund_code)
            
            # 获取基金经理数据
            self.get_fund_manager_data(fund_code)

            # 新增：获取持仓数据
            self.get_fund_holdings_data(fund_code)
            
        self._log("-" * 25)
        self._log("所有基金数据已获取完毕。")
    
    def generate_report(self):
        """生成并打印综合分析报告"""
        results = []
        for fund_code, fund_data in self.fund_data.items():
            fund_info = self.fund_info[self.fund_info['fund_code'] == int(fund_code)].iloc[0]
            fund_name = fund_info['fund_name']
            
            manager_data = self.manager_data.get(fund_code, {})
            holdings_data = self.holdings_data.get(fund_code, {})

            decision = "建议关注"
            if fund_data.get('sharpe_ratio', 0) > 1 and fund_data.get('max_drawdown', 0) < 0.2:
                decision = "高分推荐"
            elif fund_data.get('sharpe_ratio', 0) < 0.5:
                decision = "慎重考虑"
            
            results.append({
                'fund_code': fund_code,
                'fund_name': fund_name,
                'rose_3y': fund_info.get('rose_3y'),
                'rank_r_3y': fund_info.get('rank_r_3y'),
                'sharpe_ratio': fund_data.get('sharpe_ratio'),
                'max_drawdown': fund_data.get('max_drawdown'),
                'manager_name': manager_data.get('name', 'N/A'),
                'manager_return': manager_data.get('cumulative_return', np.nan),
                'tenure_years': manager_data.get('tenure_years', np.nan),
                'top_10_holdings': holdings_data.get('top_10_holdings'),
                'industry_distribution': holdings_data.get('industry_distribution'),
                'decision': decision
            })

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 50)
        print("基金综合分析报告".center(48))
        print("=" * 50)
        print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"市场趋势: {self.market_data.get('trend', 'unknown')}")
        
        print("\n所有基金分析结果:")
        print(results_df[['fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'sharpe_ratio', 'max_drawdown', 'manager_name', 'decision']].to_string(index=False))

        top_funds = results_df[results_df['rank_r_3y'] < 0.01].sort_values('rank_r_3y')
        if not top_funds.empty:
            print("\n--- 推荐基金（3年排名前 1%）---")
            print(top_funds[['fund_code', 'fund_name', 'rose_3y', 'rank_r_3y', 'sharpe_ratio', 'max_drawdown', 'manager_name', 'manager_return', 'tenure_years', 'decision']].to_string(index=False))
        else:
            print("\n没有基金满足 3 年排名前 1% 的条件。")
        print("-" * 25)
        
        # 新增：输出持仓报告
        for result in results:
            if result.get('top_10_holdings'):
                print(f"\n--- 基金 {result['fund_code']} ({result['fund_name']}) 前十持仓 ---")
                holdings_df = pd.DataFrame(result['top_10_holdings'])
                print(holdings_df.to_string(index=False))
            
            if result.get('industry_distribution'):
                print(f"\n--- 基金 {result['fund_code']} ({result['fund_name']}) 行业分布分析 ---")
                total_proportion = sum(result['industry_distribution'].values())
                print(f"行业总持仓比例: {total_proportion:.2f}%")
                
                # 按比例降序排列并打印
                sorted_industries = sorted(result['industry_distribution'].items(), key=lambda item: item[1], reverse=True)
                for industry, proportion in sorted_industries:
                    print(f"- {industry}: {proportion:.2f}%")

        print("\n" + "=" * 50)
        print("报告结束".center(48))
        print("=" * 50)
        self.analysis_report.append("\n报告生成完毕。")
        return self.analysis_report

def main():
    analyzer = FundAnalyzer()
    analyzer.load_funds_from_csv('fund_list.csv')
    analyzer.analyze_funds()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
