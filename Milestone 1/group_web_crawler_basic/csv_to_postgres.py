import psycopg2
import csv
import datetime
#
# conn = psycopg2.connect("host='localhost' port=5433 dbname='mhafizhussin' user='postgres' password='postgres'")
# cur = conn.cursor()
#
# with open('klsedata_20190225_144451.csv', 'r') as f:
#     next(f) #skip the header
#     cur.copy_from(f, 'stocks', sep=',')
# conn.commit()

# with open('user_accounts.csv', 'r') as f:
#     # Notice that we don't need the `csv` module.
#     next(f)  # Skip the header row.
#     cur.copy_from(f, 'users', sep=',')
#
# conn.commit()

# with open('klsedata_20190225_144451.csv', 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # Skip the header row.
#     for row in reader:
#         print row[1]

from numpy import genfromtxt
from time import time
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, Integer, Sequence
from sqlalchemy.engine.url import URL
import settings

# def Load_Data(file_name):
#     data = genfromtxt(file_name, delimiter=',', skip_header=1, converters={0: lambda s: str(s)})
#     return data.tolist()

Base = declarative_base()

class Price_History(Base):
    __tablename__ = "stocks"
    id = Column(Integer, Sequence('trigger_id_seq'), primary_key=True)
    thedate = Column('thedate', DateTime)
    # compname = Column('compname', String)
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

if __name__ == "__main__":
    t = time()

    #Create the database
    db_url = URL(**settings.DATABASE)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    #Create the session
    session = sessionmaker()
    session.configure(bind=engine)
    s = session()

    try:
        with open('klsedata_day20190226_153016.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row.
            for i in reader:
                record = Price_History(**{
                    'id': i[0],
                    'thedate': "2019-2-26",
                    'comp_code': i[3],
                    'open': i[4],
                    'low' : i[5],
                    'high' : i[6],
                    'lastdone': i[7],
                    'chg': i[8],
                    'chgPercent': i[9],
                    'vol': i[10],
                    'buy': i[11],
                    'sell': i[12],
                    'crawl_Timestamp': "2019-2-26"
                })
                s.add(record)
                print "ok"
        s.commit() #Attempt to commit all the records
    except:
        s.rollback() #Rollback the changes on error
    finally:
        s.close() #Close the connection
    print "Time elapsed: " + str(time() - t) + " s." #0.091s