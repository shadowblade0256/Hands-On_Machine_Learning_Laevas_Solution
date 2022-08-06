import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
LIFESAT_PATH = "datasets/lifesat"
OECD_URL = DOWNLOAD_ROOT + LIFESAT_PATH + "/oecd_bli_2015.csv"
GDP_URL = DOWNLOAD_ROOT + LIFESAT_PATH + "/gdp_per_capita.csv"

def get_oecd(oecd_url=OECD_URL,lifesat_path=LIFESAT_PATH):
    if not os.path.isdir(lifesat_path):
        os.makedirs(lifesat_path)
    oecd_path = os.path.join(lifesat_path,'oecd_bli_2015.csv')
    urllib.request.urlretrieve(oecd_url,oecd_path)

def get_gdp(gdp_url=GDP_URL,lifesat_path=LIFESAT_PATH):
    if not os.path.isdir(lifesat_path):
        os.makedirs(lifesat_path)
    gdp_path = os.path.join(lifesat_path,'gdp_per_capita.csv')
    urllib.request.urlretrieve(gdp_url,gdp_path)