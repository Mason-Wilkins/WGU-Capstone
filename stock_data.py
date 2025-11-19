import os

def download_stocks_data():
    os.system('curl -L -o stocks.sql "https://drive.usercontent.google.com/download?export=download&id=1_d8Ivcjw5WsGcUyR8C9d6Uyi-OwPumN3&confirm=t"')
    
if __name__ == "__main__":
    download_stocks_data()