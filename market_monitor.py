```python
import pandas as pd
import numpy as np
import akshare as ak
import re
import os
import logging
from datetime import datetime

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
        pattern = r'\| *(\d{6}) *\|.*?\| *(\d+\.?\d*) *\|'
        matches = re.findall(pattern, content)
        self.fund_codes = [code for code, score in matches if float(score) > 30][:20]  # 限制前20个
        logger.info("提取到 %d 个推荐基金 (限制前20): %s", len(self.fund_codes), self.fund_codes)

    def _calculate_rsi(self, returns, period=14):
        """计算14日RSI"""
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_fund_data(self, fund_code):
        """获取基金净值历史并计算技术指标"""
        try:
            logger.info("获取基金 %s 的净值数据...", fund_code)
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")  # 修正为 symbol
            df['date'] = pd.to_datetime(df['净值日期'])
            df = df.sort_values('date').tail(200)  # 取最近200个交易日
            df['returns'] = df['累计净值'].pct_change()
            
            # 计算技术指标
            df['ma50'] = df['累计净值'].rolling(50).mean()
            df['ma200'] = df['累计净值'].rolling(200).mean()
            df['rsi'] = self._calculate_rsi(df['returns'])
            ma_ratio = df['累计净值'].iloc[-1] / df['ma50'].iloc[-1] if df['ma50'].iloc[-1] else np.nan
            latest_rsi = df['rsi'].iloc[-1]
            
            self.fund_data[fund_code] = {
                'nav': df['累计净值'].iloc[-1],
                'rsi': latest_rsi,
                'ma_ratio': ma_ratio,
                'history': df[['date', '累计净值']].tail(60)  # 最近60天用于趋势图
            }
            logger.info("基金 %s 技术指标: RSI=%.2f, 净值/MA50=%.2f", fund_code, latest_rsi, ma_ratio)
        except Exception as e:
            logger.error("获取基金 %s 数据失败: %s", fund_code, str(e))
            self.fund_data[fund_code] = None

    def _get_market_sentiment(self):
        """获取市场情绪（基于上证指数）"""
        try:
            logger.info("正在获取市场情绪数据...")
            df = ak.stock_zh_index_daily_em(symbol="sh000001")
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').tail(20)
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['volume_change'] = df['volume'].pct_change()
            
            latest_ma5 = df['ma5'].iloc[-1]
            latest_ma20 = df['ma20'].iloc[-1]
            volume_trend = df['volume_change'].mean()
            
            if latest_ma5 > latest_ma20 and volume_trend > 0:
                sentiment = 'bullish'
                score = 5
            elif latest_ma5 < latest_ma20 and volume_trend < 0:
                sentiment = 'bearish'
                score = -5
            else:
                sentiment = 'neutral'
                score = 0
                
            self.market_data = {'sentiment': sentiment, 'score': score}
            logger.info("市场情绪: %s, 分数: %d", sentiment, score)
        except Exception as e:
            logger.error("获取市场情绪失败: %s", str(e))
            self.market_data = {'sentiment': 'neutral', 'score': 0}

    def _generate_trend_chart(self):
        """生成Chart.js趋势图（净值和RSI）"""
        chart_data = {
            "type": "line",
            "data": {"labels": [], "datasets": []},
            "options": {
                "title": {"display": True, "text": "基金净值和RSI趋势"},
                "scales": {
                    "yAxes": [
                        {"scaleLabel": {"display": True, "labelString": "单位净值"}},
                        {"position": "right", "scaleLabel": {"display": True, "labelString": "RSI"}}
                    ],
                    "xAxes": [{"scaleLabel": {"display": True, "labelString": "日期"}}]
                }
            }
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, fund_code in enumerate(self.fund_codes[:5]):  # 限制5个基金避免图表过密
            if fund_code in self.fund_data and self.fund_data[fund_code]:
                df = self.fund_data[fund_code]['history']
                dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
                navs = df['累计净值'].tolist()
                rsi = self._calculate_rsi(df['累计净值'].pct_change()).tail(60).tolist()
                
                chart_data['data']['labels'] = dates
                chart_data['data']['datasets'].extend([
                    {
                        "label": f"{fund_code} 净值",
                        "data": navs,
                        "borderColor": colors[i % len(colors)],
                        "fill": false,
                        "yAxisID": "yAxes[0]"
                    },
                    {
                        "label": f"{fund_code} RSI",
                        "data": rsi,
                        "borderColor": colors[i % len(colors)],
                        "borderDash": [5, 5],
                        "fill": false,
                        "yAxisID": "yAxes[1]"
                    }
                ])
        
        return chart_data

    def run(self):
        """主运行逻辑"""
        self._parse_report()
        self._get_market_sentiment()
        
        for fund_code in self.fund_codes:
            self._get_fund_data(fund_code)
        
        # 生成报告
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 市场情绪与技术指标监控报告\n\n")
            f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 市场情绪\n- 情绪: {self.market_data['sentiment']}\n- 分数: {self.market_data['score']}\n\n")
            
            f.write("## 推荐基金技术指标\n")
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
                    f.write(f"| {fund_code} | {data['nav']:.3f} | {rsi:.2f} | {ma_ratio:.2f} | {advice} |\n")
            
            f.write("\n## 趋势图\n")
            chart = self._generate_trend_chart()
            f.write(f"```chartjs\n{chart}\n```")
        
        logger.info("报告生成完成: %s", self.output_file)

if __name__ == "__main__":
    monitor = MarketMonitor()
    monitor.run()
```
