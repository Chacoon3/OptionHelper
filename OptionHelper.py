import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, brentq

__secondInYear = 31536000
def __convertT(T):
    if type(T) == datetime.datetime or type(T) == datetime.date or type(T) == np.datetime64:
        raise ValueError("T must be a number of years or timedelta")
    elif type(T) == datetime.timedelta:
        T = T.total_seconds() / __secondInYear
    T = max(T, 1 / __secondInYear)
    return T

def ivGuess(s, c, T):
    return (2 * np.pi / T) ** 0.5 * c / s

def errorCheck():
    raise ValueError("This is an error")    # python errors can be captured by dot net caller

def bsDelta(S, K, T, r, sigma, option_type='call'):
    T = __convertT(T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'call':
        delta = norm.cdf(d1)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return delta


def bsCall(S, K, T, r, sigma):    
    T = __convertT(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def bsPut(S, K, T, r, sigma):
    T = __convertT(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price


def bsGamma(S, K, T, r, sigma):
    T = __convertT(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def bsVega(S, K, T, r, sigma):
    T = __convertT(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) 
    return vega / 100


def bsIv(S, K, T, r, market_price, option_type):
    if market_price == np.nan:
        raise ValueError("Market price must be a number")
    T = __convertT(T)

    if option_type == 'call':
        def objective_function(sigma):
            return bsCall(S, K, T, r, sigma) - market_price
    elif option_type == 'put':
        def objective_function(sigma):
            return bsPut(S, K, T, r, sigma) - market_price
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    try:
        iv = newton(objective_function, maxiter= 100, x0=ivGuess(S, market_price, T))
    except:
        try:
            iv = brentq(objective_function, 1e-6, 1000)    
        except:
            iv = 10 # serves as an fallback to represent a very high IV
    return iv

def bsRho(S, K, T, r, sigma, option_type='call'):
    T = __convertT(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    
    return rho / 100


def bsTheta(S, K, T, r, sigma, option_type='call'):
    T = __convertT(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    return theta


def bsVomma(S, K, T, r, sigma):
    T = __convertT(T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vomma = S * np.sqrt(T) * norm.pdf(d1) * d1 * d2 / sigma
    return vomma


import regex as re
class OptionSymbol:

    def getSymbolString(stock_symbol,option_type,  strike_price, expiration_date:datetime.datetime):

        # Convert expiration date to YYMMDD format
        exp_date = expiration_date.strftime("%y%m%d")
        
        # Convert strike price to a string without decimal points, ensuring it has 8 characters
        strike_price_str = f"{int(strike_price * 1000):08d}"

        # Combine all parts to form the option symbol
        option_symbol = f"{stock_symbol.upper()}{exp_date}{option_type[0].upper()}{strike_price_str}"
        
        return option_symbol


    def __init__(self, symbol, underlying, strike, expiration:datetime.datetime, optionType):
        self.symbol:str = symbol
        self.underlying:str = underlying
        self.strike:float = strike
        self.expiration:datetime.datetime = expiration
        self.optionType:str = optionType

    def __str__(self):
        return self.getSymbolString(self.underlying, self.expiration, self.optionType, self.strike)


    def  __dict__(self):
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "strike": self.strike,
            "expiration": self.expiration,
            "optionType": self.optionType
        }
    

    @staticmethod
    def fromSymbolString(option_symbol_str):
        """
        Parse an option symbol string into an OptionSymbol object using regular expressions.

        Parameters:
        option_symbol_str (str): The option symbol string to parse.

        Returns:
        OptionSymbol: An instance of OptionSymbol.
        """

        pattern = re.compile(r'(?P<stock_symbol>[A-Z]{1,4})(?P<exp_date>\d{6})(?P<option_type>[CP])(?P<strike_price>\d{8})')
        match = pattern.match(option_symbol_str)
        
        if not match:
            raise ValueError("Invalid option symbol string format")

        stock_symbol = match.group('stock_symbol')
        exp_date_str = match.group('exp_date')
        expirationTime = datetime.datetime.strptime(exp_date_str, "%y%m%d")
        expirationTime = expirationTime.replace(hour=16, minute=0)
        option_type = "call" if match.group('option_type') == 'C' else "put"
        strike_price = int(match.group('strike_price')) / 1000.0

        return OptionSymbol(option_symbol_str, stock_symbol, strike_price, expirationTime, option_type, )