"""
Module for the limits determination using likelihood in the covariance matrix representation.

Description of the likelihood covariance representation: https://arxiv.org/pdf/2307.04007
Description of the limit setting procedure: https://arxiv.org/abs/1007.1727
HEPData entry with the CMS mono-V analysis results: https://www.hepdata.net/record/ins1894408
"""

import numpy as np
import pickle
from iminuit import Minuit
import scipy
import matplotlib.pyplot as plt
import os

class Limits:
    """
    Class for calculating limits using the likelihood in the covariance matrix representation.

    Attributes:
    -----------
    C: ndarray
        covariance matrix of the measurements
    m: ndarray
        measurement vector - the observed data yields
    t0: ndarray
        background-only prediction - the expected yields under the alternative hypothesis (theta = 0)
    h: ndarray
        signal template - the expected signal yields under the theta = 1 hypothesis
    outdir: str
        output directory for the plots and results
        Default is "./limits".
    """
    def __init__(self, C = None, m = None, t0 = None, h = None, outdir = "./limits"):
        self.C = C
        self.m = m
        self.t0 = t0
        self.h = h
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def set_cms_inputs(self, pickle_file):
        """
        Load the inputs from the pickle file and set the attributes C, m and t0.

        Parameters:
        -----------
        pickle_file: str
            path to the pickle file containing the dictionary with keys 'C', 'm' and 't0',
            storing the covariance matrix, data yields and background prediction, respectively.
        """
        with open(pickle_file, "rb") as f:
            inputs = pickle.load(f)
        self.C = inputs['C']
        self.m = inputs['m']
        self.t0 = inputs['t0']

    def data_yields(self, asimov = False):
        """
        Return the data yields to be used in the likelihood calculation.

        Parameters:
        -----------
        asimov: bool
            if True, the Asimov dataset (m = t0) will be returned, otherwise the observed data yields (m) will be returned.
        """
        if not asimov:
            return self.m
        else:
            return self.t0

    def nll_factory(self, asimov = False):
        """
        Factory method to create the negative log-likelihood function.
        
        The negative log-likelihood function is -2 * log L(theta) = (m - (t0 + h * theta))^T * C^-1 * (m - (t0 + h * theta)), where m is the data yields, t0 is the background prediction, h is the signal template and C is the covariance matrix.

        Parameters:
        -----------
        h: ndarray
            signal template - the expected signal yields under the theta = 1 hypothesis
        asimov: bool
            see the data_yields method
        """

        m = self.data_yields(asimov)

        def nll(theta):
            d = np.array(m) - (self.t0 + self.h * theta)
            return d.T @ np.linalg.inv(self.C) @ d
            
        return nll
    
    def find_minimum(self, nll, initial_theta):
        """
        Find the minimum of the negative log-likelihood function using the Minuit minimizer.
        
        Parameters:
        -----------
        nll: function
            the negative log-likelihood function to be minimized.
            It is a function of a single variable theta, which is the signal strength parameter.
        initial_theta: float
            the initial value of theta to start the minimization from
        """

        minuit = Minuit(nll, np.array([initial_theta]))
        minuit.migrad()
        minuit.hesse()

        # Return the value at the minimum and the uncertainty on that value.
        return minuit.values[0], minuit.errors[0]

    def theta_uncertainty(self, nll, theta):
        """
        Calculate the uncertainty on theta at a given value of theta using the curvature of the negative log-likelihood function.
        
        Parameters:
        -----------
        nll: function
            the negative log-likelihood function
        theta: float
            the value of theta at which to calculate the uncertainty
        """
        minuit = Minuit(nll, np.array([theta]))
        minuit.hesse()
        return minuit.errors[0]

    def test_statistic(self, theta, asimov = False):
        """
        Calculate the profile likelihood ratio test statistic value for given theta.

        The test statistic is defined as q(theta) = -2 * log (L(theta) / L(theta_hat)),
        where L(theta) is the likelihood at the given value of theta (the profile likelihood).
        Note that the likelihood in covariance representation is equal to the profile likelihood in the nuisance parameter representation, as discussed in the paper https://arxiv.org/abs/2307.04007.
        L(theta_hat) is the likelihood at the best-fit value of theta (theta_hat).

        Parameters:
        -----------
        theta: float
            the hypothesized value of theta
        asimov: bool
            see the data_yields method
        """
        nll = self.nll_factory(asimov)
        nll_profile = nll(theta)
        theta_hat, _ = self.find_minimum(nll, initial_theta = theta)
        nll_global = nll(theta_hat)
        return nll_profile - nll_global

    def p_value(self, theta, asimov = False):
        """
        Calculate the p-value.
        
        The test statistic distribution under the background-only hypothesis is a chi2 distribution with 1 degree of freedom, according to the paper https://arxiv.org/abs/1007.1727.

        Parameters:
        -----------
        theta: float
            the hypothesized value of theta
        asimov: bool
            see the data_yields method
        """
        ts_obs = self.test_statistic(theta, asimov)
        return scipy.stats.chi2.sf(ts_obs, df = 1)
    
    def non_centrality_parameter(self, theta, asimov = False):
        """
        Non-centrality parameter for the test statistic distribution under the background-only hypothesis.

        See the cls_value method documentation for more details.
        """

        # calculate the sigma for the non-centrality parameter
        nll = self.nll_factory(asimov)
        _, sigma = self.find_minimum(nll, initial_theta = theta)

        # non-centrality parameter for the background-only hypothesis
        return theta ** 2 / sigma ** 2

    
    def cls_value(self, theta, asimov = False, n_sigma = None):
        """
        Calculate the CLs value.

        The test statistic distribution under the background-only hypothesis is a non-central chi2 (NC chi2) distribution according to the paper https://arxiv.org/abs/1007.1727.
        The needed variance of the signal strength estimator (sigma^2)
        should be evaluated at the point theta = 0.
        However, it was checked that 0 is the only value for which the corresponding NC chi2
        distribution does not describe the actual distribution (obtained using pseudo-experiments).
        Actually, ~any value larger than 0 can be used,
        as the resulting variance is the same for all theta > 0.
        This function uses the theta_hat value to evaluate the sigma parameter.

        Parameters:
        -----------
        theta: float
            the hypothesized value of theta
        asimov: bool
            see the data_yields method
        n_sigma: float or None
            Number of standard deviations for expected limits.
            This parameter is ignored if asimov is False.
            If None, calculate the expected CLs.
            If a number, calculate the expected CLs for that number of standard deviations.
        """
        ts = self.test_statistic(theta, asimov)
        nc = self.non_centrality_parameter(theta, asimov)

        # shift the observed test statistic if when using the Asimov dataset
        # and n_sigma is not None
        if asimov and n_sigma is not None:
            # For expected limits, we evaluate the median of the background-only test statistic distribution, and the quantiles corresponding to the n_sigma standard deviations, as the observed test statistic value.
            probability = scipy.stats.norm.cdf(n_sigma)
            ts = scipy.stats.ncx2.ppf(probability, df = 1, nc = nc)

        # p-values
        p_bkg = scipy.stats.ncx2.sf(ts, df = 1, nc = nc)
        p_sig = scipy.stats.chi2.sf(ts, df = 1)

        return p_sig / p_bkg
    
    
    def p_bkg_value(self, theta, asimov = False):
        """
        Calculate the p-value from the "denominator of the CLs method".

        It is the integral from the observed test statistic value to infinity of the test statistic distribution under the background-only hypothesis.

        Parameters:
        -----------
        theta: float
            the hypothesized value of theta
        asimov: bool
            see the data_yields method
        """
        ts = self.test_statistic(theta, asimov)
        nc = self.non_centrality_parameter(theta, asimov)
        return scipy.stats.ncx2.sf(ts, df = 1, nc = nc)
    
    def find_upper_limit(self, theta_values, cls_values, cl = 95):
        """
        Find the upper limit on theta at the predefined confidence level.

        Parameters:
        -----------
        theta_values: array-like
            the values of theta to scan over.
        cls_values: array-like
            the corresponding CLs values.
        cl: float
            the confidence level for which to calculate the upper limit.
            It is in percents, so it should be between 0 and 100.
            The type I error alpha is calculated as 1 - cl / 100.
            Default cl is 95, corresponding to the 95% confidence level,
            and the upper limit is calculated as the value of theta
            for which the CLs value is equal to 0.05.

        Returns:
        --------
        upper_limit: float or None
            the upper limit on theta at the predefined confidence level.
        """
        upper_limit = None
        i = len(theta_values) - 1
        while i > 0 and cls_values[i] < 0.05:
            if cls_values[i - 1] > 0.05:
                # Perform the linear interpolation.
                theta1 = theta_values[i - 1]
                theta2 = theta_values[i]
                cls1 = cls_values[i - 1]
                cls2 = cls_values[i]
                upper_limit = theta1 + (0.05 - cls1) * (theta2 - theta1) / (cls2 - cls1)
            i -= 1
        return upper_limit


    def limits(self, theta_values = None, verbose = False):
        """
        Scan over the values of theta and calculate the p-values for each of them.

        The CLs, p, and p_bkg values are calculated for each value of theta in the theta_values array.
        They are plotted as a function of theta.
        Then, the expected CLs values and their +-1sigma, +-2sigma bands are calculated.
        They are also plotted as a function of theta, together with the observed CLs values.
        Finally, the upper limit on theta at 95% confidence level is calculated as the value of theta for which the CLs value is equal to 0.05.
        The same is done for the expected limits and their bands.

        Parameters:
        -----------
        theta_values: array-like
            the values of theta to scan over.
            If None, a default array of 20 equidistant values from 0 to 1 is used.
        verbose: bool
            if True, print the calculated limits
        """

        if theta_values is None:
            theta_values = np.linspace(0, 1, 20)

        # Theta scan with the observed data.
        cls_values = []
        p_values = []
        p_bkg_values = []
        for theta in theta_values:
            cls_values.append(self.cls_value(theta))
            p_values.append(self.p_value(theta))
            p_bkg_values.append(self.p_bkg_value(theta))

        # Plot the p-value as a function of theta:
        plt.plot(theta_values, p_values, marker='o', label='p')
        plt.plot(theta_values, p_bkg_values, marker='s', label='p_bkg')
        plt.plot(theta_values, cls_values, marker='^', label='CLs')
        plt.axhline(0.05, color='red', linestyle='dashed', label='p-value = 0.05')
        plt.xlabel(r'$\theta$')
        plt.ylabel('p-value')
        plt.title('p-value as a function of theta')
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f'{self.outdir}/theta_scan_0.png')
        plt.close()

        # Expected limits and their bands.
        exp_cls_values = {n_sigma: [] for n_sigma in [-2, -1, 0, 1, 2]}
        for theta in theta_values:
            for n_sigma in [-2, -1, 0, 1, 2]:
                cls_value = self.cls_value(theta, asimov = True, n_sigma = n_sigma)
                exp_cls_values[n_sigma].append(cls_value)      

        # Draw the expected limit as a black dashed line.
        # Draw the +- 1 sigma and +/- 2 sigma expected limits as green and yellow bands, respectively.
        # Draw the observed CLs values as a black solid line.
        plt.plot(theta_values, exp_cls_values[0], color='black', linestyle='dashed', label='Exp.')
        plt.fill_between(theta_values, exp_cls_values[-1], exp_cls_values[1], color='green', alpha=0.5, label='Exp. ± 1σ')
        plt.fill_between(theta_values, exp_cls_values[-2], exp_cls_values[2], color='yellow', alpha=0.5, label='Exp. ± 2σ')
        plt.plot(theta_values, cls_values, color='black', label='Obs.')
        plt.axhline(0.05, color='red', linestyle='dashed', label='p = 0.05')
        plt.xlabel(r'$\theta$')
        plt.ylabel('CLs')
        plt.title('CLs as a function of theta')
        plt.legend()
        plt.savefig(f'{self.outdir}/theta_scan_1.png')
        plt.close()

        # Calculate the upper limit on theta at 95% confidence level.
        obs_limit = self.find_upper_limit(theta_values, cls_values)
        exp_limits = {n_sigma: self.find_upper_limit(theta_values, exp_cls_values[n_sigma]) for n_sigma in exp_cls_values}

        # Pickle the results to a file.
        results = {
            'theta_values': theta_values,
            'cls_values': cls_values,
            'p_values': p_values,
            'p_bkg_values': p_bkg_values,
            'exp_cls_values': exp_cls_values,
            'obs_limit': obs_limit,
            'exp_limits': exp_limits,
        }
        with open(f'{self.outdir}/results.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Print the results if verbose is True.
        if verbose:
            print(f'Observed upper limit on theta at 95% CL: {obs_limit}')
            for n_sigma in exp_limits:
                print(f'Expected upper limit on theta at 95% CL for n_sigma = {n_sigma}: {exp_limits[n_sigma]}')

        return results
    
if __name__ == "__main__":
    h = np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 2, 4, 6, 8, 10, 12, 14, 2, 4, 6, 8, 10, 12, 14])
    limits = Limits(h = h)
    limits.set_cms_inputs("data/cms_monov_inputs.pkl")
    results = limits.limits(verbose = True)
