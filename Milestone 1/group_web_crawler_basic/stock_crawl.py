from lxml import html
import requests
import datetime
import time
# from time import time

now=datetime.datetime.now()
print now.strftime("%Y-%m-%d %H:%M")

# postgrest
from sqlalchemy import create_engine
from sqlalchemy import Column, String, DateTime, Integer, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import URL
import logging
import settings

klseData = [];

#KLSE Data class
class KLSEData:
    #Constructor
    def __init__(self,
                 date, compname, compCode, openTxt, lowTxt,
                 highTxt, lastDoneTxt, chgTxt, chgPercentTxt, volTxt,
                 buyTxt, sellTxt):
        self.date = date
        self.compname = compname
        self.compCode = compCode
        self.openTxt = openTxt
        self.lowTxt = lowTxt
        self.highTxt = highTxt
        self.lastDoneTxt = lastDoneTxt
        self.chgTxt = chgTxt
        self.chgPercentTxt = chgPercentTxt
        self.volTxt = volTxt
        self.buyTxt = buyTxt
        self.sellTxt = sellTxt

    # To return entire info of the class
    def toString(self):
        return '"' + self.date + '","' + self.compname + '","' + self.compCode + '","' + self.openTxt + '","' + self.lowTxt + '","' \
               + self.highTxt + '","' + self.lastDoneTxt + '","' + self.chgTxt + '","' + self.chgPercentTxt + '","' + self.volTxt + '","' \
               + self.buyTxt + '","' + self.sellTxt + '"'

# Handling the DB transaction
DeclarativeBase = declarative_base()

class Stocks(DeclarativeBase):
    __tablename__ = "stocks"
    id = Column(Integer, Sequence('trigger_id_seq'), primary_key=True)
    thedate = Column('thedate', DateTime)
    comp_code = Column('comp_code', String)
    open = Column('open', String)
    low = Column('low', String)
    high = Column('high', String)
    lastdone = Column('lastdone', String)
    chg = Column('chg', String)
    chgPercent = Column('chgPercent', String)
    vol = Column('vol', String)
    buy = Column('buy', String)
    sell = Column('sell', String)
    crawl_Timestamp = Column('crawl_Timestamp', DateTime)

db_url = URL(**settings.DATABASE)
logging.info("Creating as SQLdatabase connection at URL at URL '{db_url}'".format(db_url=db_url))

db = create_engine(db_url)
Session = sessionmaker(db)
session = Session()

DeclarativeBase.metadata.create_all(db)

def insertDataDB(data):
    psql = Stocks(
        thedate= data.date, #compname=data.compname,
        comp_code= data.compCode, open=data.openTxt,
        low=data.lowTxt, high=data.highTxt, lastdone=data.lastDoneTxt, chg=data.chgTxt,
        chgPercent=data.chgPercentTxt,
        vol=data.volTxt, buy=data.buyTxt, sell=data.sellTxt,
        crawl_Timestamp=now.strftime("%Y-%m-%d %H:%M")
    )
    session.add(psql)
    session.commit()

# KlSE Crawler main class
class KLSECrawler:
    def __init__(self, starting_url, param, depth):
        self.starting_url = starting_url
        self.depth = depth
        self.param = param
        self.apps = []

    def crawl(self):
        self.get_app_from_link(self.starting_url, self.param )
        return

    def get_app_from_link(self, link, param):

        print(link+param)
        start_page = requests.get(link+param)
        tree = html.fromstring(start_page.text)

        try:
            date = tree.xpath('//*[@id="slcontent_0_ileft_0_datetxt"]/text()')[0]
            print(date)
        except NameError:
            print("Variable x is not defined")
        except:
            return


        compname = tree.xpath('//*[@id="slcontent_0_ileft_0_compnametxt"]/text()')[0]
        openTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_opentext"]/text()')[0]
        lowTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_lowtext"]/text()')[0]
        highTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_hightext"]/text()')[0]
        lastDoneTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_lastdonetext"]/text()')[0]

        tmpChgTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_chgtext"]/text()')
        if len(tmpChgTxt) == 0 :
            tmpChgTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_chgtext"]/span/text()')

        chgTxt = tmpChgTxt[0]

        chgPercentTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_chgpercenttrext"]/text()')[0]
        volTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_voltext"]/text()')[0]
        buyTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_buyvol"]/text()')[0]
        sellTxt = tree.xpath('//*[@id="slcontent_0_ileft_0_sellvol"]/text()')[0]

        klseData.append(KLSEData(date[10:-2], compname, param, openTxt, lowTxt,
                 highTxt, lastDoneTxt, chgTxt, chgPercentTxt, volTxt,
                 buyTxt, sellTxt))

        return

# Main process start here:

#compCodeFile = open('compcode.txt', 'r')
# Get the company code for generating the URL

with open('compcode.txt') as f:
    alist = [line.rstrip() for line in f]

alist


# Crawl the data
for x in alist:

    if len(x) == 0:
        continue

    x = x.rstrip('\n')
    x = str.replace(x, '&', '%26')
    url = "https://www.thestar.com.my/business/marketwatch/stocks/?qcounter="
    print(x)
    try:
        crawler = KLSECrawler(url, x, 0)
        crawler.crawl()
    except IndexError:
        del crawler
        continue
    del crawler

# print "Time elapsed: " + str(time() - t) + " s."
# Prepare s file name as exported csv file
timeStr=time.strftime("%Y%m%d_%H%M%S", time.gmtime())
print timeStr
klseOutputFile = open('klsedata_' +timeStr+ '.csv', 'w')
klseOutputFile.write("thedate,comp_name,comp_code,open,low,high,lastDone,chg,chgPercent,vol,buy,sell\n")

# After the klse data has been extracted, the process:
# 1. generate the csv file
# 2. insert into database
for data in klseData:
    # print   data
    #print(data.toString())
    #dbIns = DBHandling()
    klseOutputFile.write(data.toString()+'\n')
    #dbIns.insertIntoDB(data)
    insertDataDB(data)

print("Done process")

try:
    crawler = KLSECrawler("https://www.thestar.com.my/business/marketwatch/stocks/?qcounter=", "TATGIAP", 0)
    crawler.crawl()
except IndexError:
     del crawler
     print "ok"


# tmp code
crawler = KLSECrawler("https://www.thestar.com.my/business/marketwatch/stocks/?qcounter=","TATGIAP", 0)
crawler.crawl()
#
# print('Size list: ' + str(len(klseData)))
#
#
# insertDataDB(klseData[0])


# ""
# for data in klseData:
#     print(data.toString())
#     dbIns = DBHandling()
#     dbIns.insertIntoDB(data)
# ""

# def print_data(data):
#     print data.date
#     print data.compname
#     print data.compCode
#     print data.buyTxt
#     return
#
# for data in klseData:
#     print_data(data)
