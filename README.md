# nycdatascience-ml-stock-analysis
Linear regression on any stock.

To run as a commandline application  
  1) install the following:  `pip install beautifulsoup pandas matplotlib sklearn`  
  2) run the script passing a command line arg(pass any stock ticker, I've used Amazon's ticker below 'AMZN'):  
  `python stock_analysis.py AMZN`  
  
__Results__ : the script will out put accuracy of three models along with a graph ui showing the mean price (average of open and close), and also predict the next the mean price of the stock up to 1 month in the future
