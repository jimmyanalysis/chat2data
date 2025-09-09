import yfinance as yf
import pandas as pd
from tqdm import tqdm

# S&P 500 component stocks list (503 stock tickers including dual-listed companies)
sp500_tickers = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK.B',
    'JPM', 'WMT', 'ORCL', 'V', 'LLY', 'MA', 'NFLX', 'XOM', 'COST', 'JNJ',
    'PLTR', 'HD', 'ABBV', 'PG', 'BAC', 'CVX', 'KO', 'AMD', 'TMUS', 'GE',
    'UNH', 'CSCO', 'PM', 'WFC', 'CRM', 'MS', 'ABT', 'LIN', 'IBM', 'MCD',
    'AXP', 'BX', 'GS', 'MRK', 'DIS', 'T', 'RTX', 'PEP', 'INTU', 'UBER',
    'CAT', 'VZ', 'TMO', 'NOW', 'BA', 'BKNG', 'BLK', 'TXN', 'SCHW', 'C',
    'ISRG', 'ANET', 'SPGI', 'QCOM', 'GEV', 'AMGN', 'ACN', 'NEE', 'BSX', 'DHR',
    'ADBE', 'TJX', 'GILD', 'SYK', 'PGR', 'PFE', 'LOW', 'COF', 'HON', 'ETN',
    'MU', 'APH', 'DE', 'UNP', 'AMAT', 'KKR', 'LRCX', 'CMCSA', 'ADP', 'COP',
    'MDT', 'PANW', 'KLAC', 'ADI', 'SNPS', 'NKE', 'MO', 'INTC', 'CB', 'WELL',
    'CRWD', 'DASH', 'ICE', 'SO', 'SBUX', 'LMT', 'MMC', 'VRTX', 'CEG', 'PLD',
    'CME', 'BMY', 'AMT', 'CDNS', 'DUK', 'TT', 'DELL', 'PH', 'MCO', 'HCA',
    'WM', 'SHW', 'CVS', 'CTAS', 'ORLY', 'RCL', 'GD', 'MCK', 'NOC', 'COIN',
    'MMM', 'MDLZ', 'CI', 'AON', 'ECL', 'APO', 'TDG', 'EQIX', 'ABNB', 'MSI',
    'NEM', 'ITW', 'PNC', 'AJG', 'UPS', 'EMR', 'FI', 'RSG', 'MAR', 'USB',
    'BK', 'WMB', 'ELV', 'HWM', 'CL', 'ZTS', 'JCI', 'CSX', 'AZO', 'VST',
    'PYPL', 'EOG', 'SPG', 'APD', 'HLT', 'MNST', 'NSC', 'ADSK', 'FCX', 'FTNT',
    'TEL', 'WDAY', 'TRV', 'AEP', 'REGN', 'AXON', 'KMI', 'URI', 'CMG', 'DLR',
    'TFC', 'NXPI', 'COR', 'PWR', 'ROP', 'AFL', 'BDX', 'FAST', 'GLW', 'CARR',
    'CMI', 'ALL', 'NDAQ', 'FDX', 'GM', 'O', 'SRE', 'IDXX', 'D', 'MET',
    'PCAR', 'LHX', 'PSX', 'PSA', 'PAYX', 'DHI', 'SLB', 'MPC', 'CTVA', 'ROST',
    'AMP', 'KDP', 'CBRE', 'OKE', 'TGT', 'GWW', 'XYZ', 'KR', 'EBAY', 'EW',
    'F', 'CPRT', 'EXC', 'GRMN', 'CCI', 'AIG', 'DDOG', 'KMB', 'OXY', 'EA',
    'MSCI', 'TTWO', 'BKR', 'PEG', 'XEL', 'VLO', 'AME', 'RMD', 'YUM', 'FANG',
    'CHTR', 'KVUE', 'ETR', 'MPWR', 'CCL', 'DAL', 'ROK', 'VMC', 'SYY', 'CSGP',
    'PRU', 'TKO', 'VRSK', 'LYV', 'FIS', 'LVS', 'HIG', 'MLM', 'ED', 'HSY',
    'CAH', 'MCHP', 'TRGP', 'HUM', 'VICI', 'LEN', 'WEC', 'XYL', 'OTIS', 'CTSH',
    'ACGL', 'A', 'GEHC', 'NUE', 'PCG', 'EQT', 'EL', 'STX', 'UAL', 'KHC',
    'IQV', 'RJF', 'WAB', 'WTW', 'FICO', 'ODFL', 'TSCO', 'DXCM', 'BRO', 'STT',
    'IR', 'VTR', 'DD', 'EFX', 'BR', 'EXR', 'MTB', 'STZ', 'WBD', 'AWK',
    'NRG', 'DTE', 'ADM', 'FITB', 'KEYS', 'K', 'ROL', 'HPE', 'AEE', 'MTD',
    'AVB', 'PPL', 'IRM', 'WRB', 'SMCI', 'ATO', 'VLTO', 'SYF', 'GIS', 'FOXA',
    'WDC', 'CBOE', 'EXPE', 'PHM', 'TTD', 'FOX', 'EQR', 'VRSN', 'TDY', 'PPG',
    'FE', 'HPQ', 'WSM', 'TYL', 'CNP', 'DG', 'PTC', 'IP', 'HBAN', 'DOV',
    'ES', 'STE', 'DRI', 'NTRS', 'LULU', 'CINF', 'SBAC', 'NVR', 'DLTR', 'TROW',
    'ULTA', 'JBL', 'RF', 'HUBB', 'EXE', 'LDOS', 'CHD', 'PODD', 'CPAY', 'LH',
    'SW', 'FSLR', 'NTAP', 'CMS', 'DVN', 'EIX', 'CDW', 'GPN', 'CFG', 'LII',
    'ON', 'TPR', 'TPL', 'GDDY', 'ZBH', 'BIIB', 'TSN', 'AMCR', 'DGX', 'KEY',
    'L', 'NI', 'TRMB', 'GEN', 'ERIE', 'GPC', 'STLD', 'MKC', 'WY', 'INVH',
    'CTRA', 'IT', 'HAL', 'FFIV', 'WST', 'NWS', 'J', 'RL', 'TER', 'WAT',
    'PKG', 'PFG', 'PNR', 'ESS', 'LYB', 'SNA', 'MAA', 'INCY', 'IFF', 'NWSA',
    'DOW', 'LNT', 'EVRG', 'LUV', 'BG', 'FTV', 'EXPD', 'APTV', 'ZBRA', 'HRL',
    'MAS', 'DECK', 'DPZ', 'BLDR', 'HOLX', 'BBY', 'CLX', 'OMC', 'COO', 'BALL',
    'KIM', 'ALLE', 'CHRW', 'BF.B', 'TXT', 'CNC', 'UDR', 'FDS', 'EG', 'CF',
    'JBHT', 'AVY', 'BEN', 'ARE', 'REG', 'SOLV', 'VTRS', 'IEX', 'BAX', 'DOC',
    'PAYC', 'NDSN', 'POOL', 'GNRC', 'SJM', 'JKHY', 'CPT', 'BXP', 'SWK', 'HAS',
    'WYNN', 'UHS', 'GL', 'HST', 'MRNA', 'SWKS', 'NCLH', 'PNW', 'AIZ', 'AKAM',
    'RVTY', 'HII', 'ALGN', 'WBA', 'MOS', 'TAP', 'AOS', 'MGM', 'CPB', 'ALB',
    'DVA', 'IPG', 'AES', 'IVZ', 'MTCH', 'CAG', 'MOH', 'EPAM', 'KMX', 'TECH',
    'DAY', 'HSIC', 'FRT', 'LKQ', 'MHK', 'CRL', 'LW', 'EMN', 'APA', 'MKTX',
    'CZR', 'ENPH'
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
        # 'PEG': info.get('pegRatio'),
        'Revenue_YoY': info.get('revenueGrowth'),
        'EPS_YoY': info.get('earningsQuarterlyGrowth'),
        'Market_Cap': info.get('marketCap')
    })

# Save data to CSV file
df = pd.DataFrame(data)
df.to_csv('sp500_financials.csv', index=False)

print("Data has been saved to sp500_financials.csv")