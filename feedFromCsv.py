import numpy as np
import pandas as pd
import datetime
# import pandas_datareader.data as web
from bs4 import BeautifulSoup
import requests


def feedFromCsv(ticker):
	df = pd.read_csv('csv_sources/{}_d.csv'.format(ticker))

	def updateStock(ticker):
		if df[-1:]['Date'].item() == str(datetime.datetime.today().date()):
			print('Data up to date')
			return pd.DataFrame()

		resp = requests.get('https://stooq.com/q/d/?s=' + ticker)
		soup = BeautifulSoup(resp.text, 'html.parser')
		table = soup.find('table', { 'id': 'fth1' })
		if not table:
			print('invalid ticker or site limit exceeded')
			return pd.DataFrame()

		rows = table.findAll('tr')[1:]
		dff = pd.DataFrame()
		for row in rows:
			rdata = row.findAll('td')
			topDate = datetime.datetime.strptime(rdata[1].text, '%d %b %Y').date()
			if df[-1:]['Date'].item() == str(topDate):
				break
			rdict = {}
			rdict['Date'] = topDate
			rdict['Open'] = float(rdata[2].text)
			rdict['High'] = float(rdata[3].text)
			rdict['Low'] = float(rdata[4].text)
			rdict['Close'] = float(rdata[5].text)
			rdict['Volume'] = int(rdata[8].text.replace(',', ''))
			dff = dff.append(rdict, ignore_index=True)

		dff = dff.iloc[::-1]
		return dff

	dfn = updateStock(ticker)
	if not dfn.empty:
		nLen = len(dfn)
		df.drop(df.index[:nLen], inplace=True)
		df = df.append(dfn, ignore_index=True)
		df.to_csv('csv_sources/{}_d.csv'.format(ticker), index=False)
		print('Data actualized')

	return df

def matrixOfReturns(tickers):
	mtx = np.empty((0, 1000))
	for ticker in tickers:
		df = feedFromCsv(ticker)
		df['dRet'] = df['Close'].pct_change() * 100
		df.dropna(inplace=True)
		mtx = np.append(mtx, np.asmatrix(df['dRet']), axis=0)

	return mtx
