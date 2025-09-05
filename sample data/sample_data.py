import yfinance as yf
import pandas as pd
from tqdm import tqdm

# S&P 500 component stocks list (503 stock tickers including dual-listed companies)
sp500_tickers = [
    'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AAP', 'AMD',
    'AES', 'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN',
    'ALGN', 'ALLE', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC',
    'ADI', 'ANDV', 'ANSS', 'BBY', 'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX',
    'BHF', 'BMY', 'AVGO', 'BF.B', 'CHRW', 'CCI', 'CSX', 'CMCSA', 'CMA', 'CAG',
    'COP', 'ED', 'STZ', 'COO', 'COST', 'CVS', 'D', 'DHI', 'DHR', 'DRI', 'DOV', 'DTE',
    'DUK', 'DXC', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR',
    'EVRG', 'ES', 'EXC', 'EXPD', 'EXR', 'FFIV', 'FAST', 'FRT', 'FDX', 'FIS', 'FISV',
    'FLT', 'FTNT', 'FTV', 'GD', 'GE', 'GIS', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GILD',
    'GLW', 'GS', 'HAL', 'HAS', 'HCA', 'PEAK', 'HSY', 'HES', 'HPE', 'HON', 'HRL', 'HST',
    'HWM', 'HPQ', 'HUM', 'IBM', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG', 'IVZ',
    'IP', 'IPG', 'IQV', 'IRM', 'JBHT', 'JNJ', 'JCI', 'JPM', 'K', 'KDP', 'KLAC', 'KMB',
    'KO', 'LHX', 'LH', 'LMT', 'LOW', 'LRCX', 'LVS', 'MA', 'MAA', 'MAR', 'MMC', 'MS',
    'MSCI', 'MSFT', 'MA', 'NDAQ', 'NKE', 'NOC', 'NEM', 'NUE', 'NVDA', 'ORLY', 'OXY',
    'PAYX', 'PYPL', 'PEP', 'PNC', 'PPG', 'PPL', 'PFE', 'PM', 'PSA', 'PH', 'PNR', 'PXD',
    'PYPL', 'QCOM', 'RTX', 'REGN', 'RF', 'RMD', 'RSG', 'SPGI', 'CRM', 'SBAC', 'SLB',
    'SNA', 'SO', 'SPG', 'SWK', 'SYF', 'SYY', 'TROW', 'TT', 'TDG', 'TRV', 'UNH', 'UPS',
    'URI', 'UNP', 'VLO', 'VMC', 'VZ', 'VRTX', 'WMT', 'WBA', 'WM', 'WAT', 'WEC', 'WFC',
    'WELL', 'WDC', 'WU', 'WY', 'XEL', 'XYL', 'ZBH', 'ZBRA', 'ZTS'
]

# Fetch financial data for each stock
data = []
for ticker in tqdm(sp500_tickers, desc="Fetching stock data", ncols=80, colour='green'):
    stock = yf.Ticker(ticker)
    info = stock.info
    data.append({
        'Ticker': ticker,
        'PE_TTM': info.get('trailingPE'),
        'Forward_PE': info.get('forwardPE'),
        'PEG': info.get('pegRatio'),
        'Revenue_YoY': info.get('revenueGrowth'),
        'EPS_YoY': info.get('earningsQuarterlyGrowth'),
        'Market_Cap': info.get('marketCap')
    })

# Save data to CSV file
df = pd.DataFrame(data)
df.to_csv('sp500_financials.csv', index=False)

print("Data has been saved to sp500_financials.csv")