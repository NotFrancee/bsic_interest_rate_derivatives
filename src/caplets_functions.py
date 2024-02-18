import numpy as np
from scipy.optimize import fmin
from scipy.stats import norm


def caplet_price(r_k, f_i, N, delta_i, T, discount_factor, sigma):
    """
    Get Caplet price given the Black inputs.

    Parameters
    ----------
    r_k : float
        Strike Reference Floating Rate.
    f_i : float
        Forward Reference Rate in period Ti-Ti+1 at T0.
    N : float
        Notional.
    delta_i : float
        Payment Frequency in years (Calendar Days).
    T : float
        Time to Expiration in years (Calendar Days).
    discount_factor : float
        Zero discount rate from T0 to Ti.
    sigma : float
        Annualized Volatility of Reference Rate.

    Returns
    -------
    float
        Price of the Caplet at T0 according to Black's Formula.
    """
    d1 = (np.log(f_i / r_k) + sigma**2 / 2 * T) / (sigma * np.sqrt(T))
    d2 = (np.log(f_i / r_k) - sigma**2 / 2 * T) / (sigma * np.sqrt(T))

    return N * delta_i * discount_factor * (f_i * norm.cdf(d1) - r_k * norm.cdf(d2))


def implied_volatility_caplet(
    quoted_price,
    r_k,
    f_i,
    N,
    delta_i,
    T,
    discount_factor,
    sigma_0=0.2,
    debug=False,
):
    """
    Get Implied Volatility from Black's formula for a given Caplet and inputs.

    Parameters
    ----------
    quoted_price : float
        The price of the caplet as quoted.
    r_k : float
        Strike Reference Floating Rate.
    f_k : float
        Forward Reference Rate in period Ti-Ti+1 at T0.
    N : float
        Notional.
    delta_i : float
        Payment Frequency in years (Calendar Days).
    T : float
        Time to Expiration in years (Calendar Days).
    discount_factor : float
        Zero discount rate from T0 to Ti.
    sigma_0 : float, optional
        Initial guess for volatility, by default 0.3.
    debug : bool, optional
        Whether to print debug info while minimizing the obj function, by default False.

    Returns
    -------
    float
        The Annualized Implied Volatility of the Caplet.

    """

    # define objective function to minimize
    # the squared error between Black's price and quoted price
    def objective_function(s):
        black_price = caplet_price(r_k, f_i, N, delta_i, T, discount_factor, s)
        val = (black_price - quoted_price) ** 2

        if debug:
            print(f"[sigma] = {s}, Objective Function Value: {val}")

        return val

    # use fmin to find the value of sigma that minimizes the obj function
    sigma = fmin(objective_function, [sigma_0], disp=debug)
    return sigma
