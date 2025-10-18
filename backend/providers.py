import aiohttp

async def fetch_ohlc_stooq(ticker: str, start: str, end: str):
    url = f"https://stooq.com/q/d/l/?s={ticker}&d1={start.replace('-','')}&d2={end.replace('-','')}&i=d"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            text = await r.text()
    lines = text.strip().split("\n")[1:]
    result = []
    for line in lines:
        d,o,h,l,c,v = line.split(",")
        result.append({"date": d, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(v)})
    return result

async def fetch_news_stub(ticker: str, start: str, end: str):
    return [{"date": start, "headline": f"No major news for {ticker} between {start} and {end}."}]
