"""
Module for the limits determination using the CMS simplified likelihood.

Description of the simplified likelihood used by CMS: https://cds.cern.ch/record/2242860
Description of the limit setting procedure: https://arxiv.org/abs/1007.1727
HEPData entry with the CMS monojet/V analysis results: https://www.hepdata.net/record/ins1894408
"""

import numpy as np
import pickle
from iminuit import Minuit
import scipy
import matplotlib.pyplot as plt
import os

class Limits:
    """
    Class for calculating limits using the simplified likelihood.

    Attributes:
    -----------
    V: ndarray
        covariance matrix of the measurements
    n: ndarray
        measurement vector - the observed data yields
    b: ndarray
        background-only prediction - the expected yields under the alternative hypothesis (mu = 0)
    s: ndarray
        signal template - the expected signal yields under the mu = 1 hypothesis
    outdir: str
        output directory for the plots and results
        Default is "./limits".
    """
    def __init__(self, V = None, n = None, b = None, s = None, outdir = "./limits"):
        self.V                      = V
        self.n                      = n
        self.b                      = b
        self.s                      = s
        self.outdir                 = outdir
        self.init_theta             = np.zeros_like(self.b)
        self.non_centrality         = 0
        self.asymptotic             = True
        self.n_pseudoexperiments    = 1000
        self.dict_pseudoexperiments = {}
        self.mu_tested              = 1
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def set_cms_inputs(self, pickle_file):
        """
        Load the inputs from the pickle file and set the attributes V, n and b.

        Parameters:
        -----------
        pickle_file: str
            path to the pickle file containing the dictionary with keys 'V', 'n' and 'b',
            storing the covariance matrix, data yields and background prediction, respectively.
        """
        with open(pickle_file, "rb") as f:
            inputs = pickle.load(f)
        self.V = inputs['V']
        self.n = inputs['n']
        self.b = inputs['b']
        self.init_theta = np.zeros_like(self.b)

    def pseudoexperiments(self, n_pseudoexperiments = 1000, read_pseudoexperiments_from_file = True, pseudoexperiments_file = "limits/dict_pseudoexperiments.pkl"):
        """
        Use pseudo-experiments to determine the distributions of the test statistic needed for the CLs method.
        This method just sets the necessary attributes to use pseudo-experiments instead of the asymptotic formulae for the test statistic distributions.

        Parameters:
        -----------
        n_pseudoexperiments: int
            The number of pseudo-experiments to generate.
        read_pseudoexperiments_from_file: bool
            If True, the pseudo-experiments will be read from a file instead of generated.
        pseudoexperiments_file: str
            The path to the file containing the pre-generated pseudo-experiments.
        """
        self.asymptotic = False
        self.n_pseudoexperiments = n_pseudoexperiments

        if read_pseudoexperiments_from_file:
            if os.path.exists(pseudoexperiments_file):
                with open(pseudoexperiments_file, "rb") as f:
                    self.dict_pseudoexperiments = pickle.load(f)
                    # Pick an arbitrary key from the dict_pseudoexperiments to set self.n_pseudoexperiments
                    _, arbitrary = list(self.dict_pseudoexperiments.items())[1]
                    self.n_pseudoexperiments = len(arbitrary)
            else:
                print(f"File {pseudoexperiments_file} not found. Generating pseudo-experiments.")
                self.dict_pseudoexperiments = {}
        return    

    def data_yields(self, asimov = False, mu = None):
        """
        Return the data yields to be used in the likelihood calculation.

        Parameters:
        -----------
        asimov: bool
            If True, the Asimov dataset will be returned, otherwise the observed data yields (n) will be returned.
        mu: float or None
            The value of the signal strength parameter to be used in the Asimov dataset.
        """
        if not asimov:
            return self.n
        else:
            if mu is None:
                raise ValueError("mu must be provided when asimov is True")
            
            nll = self.nll_factory(self.n)
            minuit = Minuit(nll, [mu] + list(self.init_theta))
            minuit.fixed[0] = True
            minuit.migrad()
            minuit.hesse()
            hat_hat_theta = np.array(minuit.values[1:])

            # Asimov dataset
            return mu * self.s + self.b + hat_hat_theta

    def nll_factory(self, n = None):
        r"""
        Factory method to create the negative log-likelihood function.
        
        The negative log-likelihood function is -2 * log L(mu, theta) = -2 \sum_i (n_i \ln(\mu s_i + b_i + \theta_i) - (\mu s_i + b_i + \theta_i)) + \sum_{ij}\theta_i V^{-1}_{ij} \theta_j, where n is the data yields, b is the prefit background prediction, s is the signal template and V^{-1} is inverse of the covariance matrix.

        Parameters:
        -----------
        n: ndarray
            The data yields to be used in the likelihood calculation.
            It can be the observed data yields or an Asimov dataset or a pseudo-experiment dataset.
            If None, the observed data yields (self.n) will be used.

        Returns:
        --------
        nll: function
            The negative log-likelihood function.
        """

        if n is None:
            n = self.n

        def nll(params):        
            mu    = params[0]
            theta = np.array(params[1:])
            t = mu * self.s + self.b + theta
            value = -2 * np.sum(n * np.log(t) - t)
            value += 2 * np.sum(scipy.special.gammaln(n + 1))
            value += theta @ np.linalg.inv(self.V) @ theta
            return value

        return nll
    
    def find_minimum(self, nll, init_mu = 0, init_theta = None, fix_mu = False):
        """
        Find the minimum of the negative log-likelihood function using the Minuit minimizer.
        
        Parameters:
        -----------
        nll: function
            The negative log-likelihood function to be minimized.
            It is a function of the signal strength mu and the nuisance parameters theta.
        init_mu: float
            The initial value of mu to start the minimization from
        init_theta: float
            The initial value of theta to start the minimization from
        fix_mu: bool
            If True, the signal strength parameter mu will be fixed to the value of init_mu during the minimization, otherwise it will be allowed to float.
        """

        if init_theta is None:
            init_theta = self.init_theta

        minuit = Minuit(nll, [init_mu] + list(init_theta))
        if fix_mu:
            minuit.fixed[0] = True
        minuit.migrad()
        minuit.hesse()

        # Return the value at the minimum and the uncertainty on that value.
        # Also return the value of the negative log-likelihood at the minimum.
        return minuit.values[0], minuit.errors[0], minuit.fval

    def test_statistic(self, nll, mu, init_theta = None):
        """
        Calculate the profile likelihood ratio test statistic value for given mu.

        The test statistic is defined as q(mu) = -2 * log (L(hat_hat_mu, hat_hat_theta) / L(hat_mu, hat_theta))
        if hat_mu <= mu, and q(mu) = 0 otherwise.

        Parameters:
        -----------
        mu: float
            The hypothesized value of mu
        init_theta: ndarray or None
            The initial values of the nuisance parameters to start the minimization from.
            If None, the default values self.init_theta will be used.
        """
        
        # Denominator.
        hat_mu, _, global_min = self.find_minimum(nll, init_mu = mu, init_theta = init_theta, fix_mu = False)

        # Return zero if mu_hat > mu.
        if hat_mu > mu:
            return 0

        # Nominator.
        _, _, constraint_min = self.find_minimum(nll, init_mu = mu, init_theta = init_theta, fix_mu = True)

        return constraint_min - global_min

    def p_value(self, test_statistic_value):
        """
        Calculate the p-value.
        
        The test statistic distribution under the background-only hypothesis is a half-chi2 distribution with 1 degree of freedom, according to the paper https://arxiv.org/abs/1007.1727.
        Its cumulative distribution is given by Eq. (58) in the same paper,
        and it is the gaussian CDF of the square root of the test statistic value.

        Parameters:
        -----------
        test_statistic_value: float
            The test statistic value for which to calculate the p-value.
        """
        if self.asymptotic:
            return scipy.stats.norm.sf(np.sqrt(test_statistic_value))
        else:
            mu = self.mu_tested
            # if (mu, mu) not in self.dict_pseudoexperiments:
            #     self.generate_pseudoexperiments(mu, mu)
            return np.sum(self.dict_pseudoexperiments[(mu, mu)] >= test_statistic_value) / self.n_pseudoexperiments

    def p_bkg_value(self, test_statistic_value):
        """
        Calculate the p-value from the "denominator of the CLs method".

        It is the integral from the test statistic value to infinity of the test statistic distribution under the background-only hypothesis.
        The cumulative distribution of the test statistic under the background-only hypothesis is given by Eq. (57) in https://arxiv.org/pdf/1007.1727.pdf.
        The non-centrality parameter of the related NC chi2 distribution is estimated as the value of the test statistic for the Asimov dataset with mu = 0.

        Parameters:
        -----------
        test_statistic_value: float
            The test statistic value for which to calculate the p-value.
        """
        if self.asymptotic:
            return scipy.stats.norm.sf(np.sqrt(test_statistic_value) - np.sqrt(self.non_centrality))
        else:
            mu = self.mu_tested
            # if (mu, 0) not in self.dict_pseudoexperiments:
            #     self.generate_pseudoexperiments(mu, 0)
            return np.sum(self.dict_pseudoexperiments[(mu, 0)] >= test_statistic_value) / self.n_pseudoexperiments
    
    def ppf_bkg(self, probability):
        """
        Calculate the quantile function (inverse CDF) of the test statistic distribution under the background-only hypothesis.

        Parameters:
        -----------
        probability: float
            The probability for which to calculate the quantile.
        non_centrality: float
            The non-centrality parameter of the asymptotic test statistic distribution under the background-only hypothesis.
            This parameter is only used when self.asymptotic is True.
        mu: float
            The tested mu value.
            Only used when self.asymptotic is False.
        """
        if self.asymptotic:
            sqrt_test_statistic_value = scipy.stats.norm.ppf(probability, loc = np.sqrt(self.non_centrality))
            if sqrt_test_statistic_value < 0:
                test_statistic_value = 0
            else:
                test_statistic_value = sqrt_test_statistic_value ** 2
            return test_statistic_value
        else:
            mu = self.mu_tested
            # if (mu, 0) not in self.dict_pseudoexperiments:
            #     self.generate_pseudoexperiments(mu, 0)
            return np.percentile(self.dict_pseudoexperiments[(mu, 0)], probability * 100)

    def cls_value(self, test_statistic_value, n_sigma = None):
        """
        Calculate the CLs value.

        Parameters:
        -----------
        test_statistic_value: float
            The test statistic value for which to calculate the CLs value.
        n_sigma: float or None
            Number of standard deviations for expected limits.
            If None, calculate the CLs using the provided test statistic value.
            Case:
              0: the median of the background-only test statistic distribution is used as the observed test statistic value.
              1: the ~0.84 quantile of the background-only test statistic distribution is used as the observed test statistic value.
              -1: the ~0.16 quantile of the background-only test statistic distribution is used as the observed test statistic value.
              etc.
        """

        if n_sigma is not None:
            # For expected limits, we evaluate the median of the background-only test statistic distribution,
            # and the quantiles corresponding to the n_sigma standard deviations,
            # as the observed test statistic value.
            probability = scipy.stats.norm.cdf(n_sigma)
            test_statistic_value = self.ppf_bkg(probability)

        # p-values
        p_bkg = self.p_bkg_value(test_statistic_value)
        p_sig = self.p_value(test_statistic_value)

        return p_sig / p_bkg
    
    def find_upper_limit(self, mu_values, cls_values, cl = 95):
        """
        Find the upper limit on mu at the predefined confidence level.

        Parameters:
        -----------
        mu_values: array-like
            the values of mu to scan over.
        cls_values: array-like
            the corresponding CLs values.
        cl: float
            the confidence level for which to calculate the upper limit.
            It is in percents, so it should be between 0 and 100.
            The type I error alpha is calculated as 1 - cl / 100.
            Default cl is 95, corresponding to the 95% confidence level,
            and the upper limit is calculated as the value of mu
            for which the CLs value is equal to 0.05.

        Returns:
        --------
        upper_limit: float or None
            The upper limit on mu at the predefined confidence level.
        """
        upper_limit = None
        i = len(mu_values) - 1
        while i > 0 and cls_values[i] < 0.05:
            if cls_values[i - 1] > 0.05:
                # Perform the linear interpolation.
                mu1 = mu_values[i - 1]
                mu2 = mu_values[i]
                cls1 = cls_values[i - 1]
                cls2 = cls_values[i]
                upper_limit = mu1 + (0.05 - cls1) * (mu2 - mu1) / (cls2 - cls1)
            i -= 1
        return upper_limit

    def limits(self, mu_values = None, verbose = False):
        """
        Scan over the values of mu and calculate the p-values for each of them.

        The CLs, p, and p_bkg values are calculated for each value of mu in the mu_values array.
        They are plotted as a function of mu.
        Then, the expected CLs values and their +-1sigma, +-2sigma bands are calculated.
        They are also plotted as a function of mu, together with the observed CLs values.
        Finally, the upper limit on mu at 95% confidence level is calculated as the value of mu for which the CLs value is equal to 0.05.
        The same is done for the expected limits and their bands.

        Parameters:
        -----------
        mu_values: array-like
            the values of mu to scan over.
            If None, a default array of 20 equidistant values from 0 to 1 is used.
        verbose: bool
            if True, print the calculated limits
        """

        if mu_values is None:
            mu_values = np.linspace(0, 10, 10)

        # mu scan with the observed data.
        nll = self.nll_factory()
        n_asimov = self.data_yields(asimov = True, mu = 0)
        nll_asimov = self.nll_factory(n_asimov)
        cls_values = []
        p_values = []
        p_bkg_values = []
        exp_cls_values = {n_sigma: [] for n_sigma in [-2, -1, 0, 1, 2]}
        exp_cls_values['asimov'] = []

        for mu in mu_values:
            ts = self.test_statistic(nll, mu)
            self.ts = ts
            self.mu_tested = mu
            self.non_centrality = self.test_statistic(nll_asimov, mu)
            if not self.asymptotic:
                self.generate_pseudoexperiments()
            cls_values.append(self.cls_value(ts))
            p_values.append(self.p_value(ts))
            p_bkg_values.append(self.p_bkg_value(ts))

            for n_sigma in [-2, -1, 0, 1, 2]:
                cls_value = self.cls_value(ts, n_sigma = n_sigma)
                exp_cls_values[n_sigma].append(cls_value)
            # The test statistic value for the Asimov dataset is equal to the non-centrality.
            exp_cls_values['asimov'].append(self.cls_value(self.non_centrality))

        # Plot the p-value as a function of mu:
        plt.plot(mu_values, p_values, marker='o', label='p')
        plt.plot(mu_values, p_bkg_values, marker='s', label='p_bkg')
        plt.plot(mu_values, cls_values, marker='^', label='CLs')
        plt.axhline(0.05, color='red', linestyle='dashed', label='p-value = 0.05')
        plt.xlabel(r'$\mu$')
        plt.ylabel('p-value')
        plt.title('p-value as a function of mu')
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f'{self.outdir}/mu_scan_0.png')
        plt.close()

        # Plot the expected limits and their bands.
        #   - Draw the expected limit as a black dashed line.
        #   - Draw the +- 1 sigma and +/- 2 sigma expected limits as green and yellow bands, respectively.
        #   - Draw the observed CLs values as a black solid line.
        plt.plot(mu_values, exp_cls_values[0], color='black', linestyle='dashed', label='Exp.')
        plt.fill_between(mu_values, exp_cls_values[-1], exp_cls_values[1], color='green', alpha=0.5, label='Exp. ± 1σ')
        plt.fill_between(mu_values, exp_cls_values[-2], exp_cls_values[2], color='yellow', alpha=0.5, label='Exp. ± 2σ')
        plt.plot(mu_values, exp_cls_values['asimov'], color='blue', linestyle='dashed', label='Exp. As.')
        plt.plot(mu_values, cls_values, color='black', label='Obs.')
        plt.axhline(0.05, color='red', linestyle='dashed', label='p = 0.05')
        plt.xlabel(r'$\mu$')
        plt.ylabel('CLs')
        plt.title('CLs as a function of mu')
        plt.legend()
        plt.savefig(f'{self.outdir}/mu_scan_1.png')
        plt.close()

        # Calculate the upper limit on theta at 95% confidence level.
        obs_limit = self.find_upper_limit(mu_values, cls_values)
        exp_limits = {n_sigma: self.find_upper_limit(mu_values, exp_cls_values[n_sigma]) for n_sigma in exp_cls_values}

        # Pickle the results to a file.
        results = {
            'mu_values': mu_values,
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
            print(f'Observed upper limit on mu at 95% CL: {obs_limit}')
            for n_sigma in exp_limits:
                print(f'Expected upper limit on mu at 95% CL for n_sigma = {n_sigma}: {exp_limits[n_sigma]}')

        return results
    
    def generate_pseudoexperiments(self):
        self._generate_pseudoexperiments(self.mu_tested, 0)
        self._generate_pseudoexperiments(self.mu_tested, self.mu_tested)
        self.plot_pseudoexperiments()
        return
    
    def _generate_pseudoexperiments(self, mu_tested, mu_pe):
        """
        Generate pseudo-experiments and calculate the test statistic value for each of them.

        The pseudo-experiments are generated as Poisson fluctuations of the expected yields nu = mu_pe * s + b + hat_hat_theta(mu_pe),
        where hat_hat_theta(mu_pe) are the values of the nuisance parameters that minimize the negative log-likelihood for the given fixed value of mu_pe.

        Parameters:
        -----------
        mu_tested: float
            The tested value of the signal strength parameter - the value with which the test statistic is calculated.
        mu_pe: float
            The value of the signal strength parameter with which the pseudo-experiments are generated.
        n_pseudoexperiments: int
            The number of pseudo-experiments to generate.
        """

        # No need to generate the pseudo-experiments if they have already been generated for the given values of mu_tested and mu_pe.
        if (mu_tested, mu_pe) in self.dict_pseudoexperiments:
            return

        nu = self.data_yields(asimov = True, mu = mu_pe)
        pseudo_experiments = np.random.poisson(nu, size=(self.n_pseudoexperiments, len(nu)))

        ts_toys = []
        for i in range(self.n_pseudoexperiments):
            pseudo_n = pseudo_experiments[i]
            nll = self.nll_factory(pseudo_n)
            ts = self.test_statistic(nll, mu_tested, self.init_theta)
            ts_toys.append(ts)

        self.dict_pseudoexperiments[(mu_tested, mu_pe)] = np.array(ts_toys)

        return
    
    def plot_pseudoexperiments(self):
        """
        Plot the generated and asymptotic distributions of the test statistic.
         
        For the signal+background, the distribution is the half chi-square distribution,
        which is a sum of two parts:
          - the delta function at zero with weight 0.5
          - the chi-square distribution with 1 degree of freedom with weight 0.5.

        For the background-only, the distribution is a non-central chi-square distribution
        with 1 degree of freedom and non-centrality parameter equal to the value
        of the test statistic for the Asimov dataset with mu = 0.
        """
        from scipy.stats import chi2, norm

        plt.figure(figsize=(10, 6))

        # Plot the signal+background distributions.
        # -----------------------------------------
        n_bins = 20
        upper_bound = 10
        bin_edges = np.linspace(0, upper_bound, n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # S+B pseudoexperiments distribution.
        h, h_unc = self.histogram_pseudoexperiments(self.dict_pseudoexperiments[(self.mu_tested, self.mu_tested)], bin_edges)
        plt.errorbar(bin_centers, h, yerr=h_unc, fmt='o', color='navy', label='s+b, p.e.')

        # S+B asymptotic distribution.
        half_chi2_hist = chi2.cdf(bin_edges[1:], df=1) - chi2.cdf(bin_edges[:-1], df=1)
        half_chi2_hist[0] += 1 # Add the delta function at zero.
        half_chi2_hist[-1] += chi2.sf(upper_bound, df=1) # Add the tail above the upper bound to the last bin.
        half_chi2_hist *= 0.5 / bin_width  # Normalize by the bin width to get the probability density. The factor of 0.5 is due to the fact that we are summing two pdfs.
        plt.stairs(half_chi2_hist, bin_edges, fill = False, label='s+b, asymp.', color='navy')

        # Plot the background-only distributions.
        # ---------------------------------------
        n_bins = 10
        upper_bound = 30
        bin_edges = np.linspace(0, upper_bound, n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # B-only pseudo-experiments distribution.
        h, h_unc = self.histogram_pseudoexperiments(self.dict_pseudoexperiments[(self.mu_tested, 0)], bin_edges)
        plt.errorbar(bin_centers, h, yerr=h_unc, fmt='o', color='red', label='b, p.e.')

        # B-only asymptotic distribution.
        shift = np.sqrt(self.non_centrality)
        half_chi2_hist = (norm.cdf(np.sqrt(bin_edges[1:]) - shift) - norm.cdf(np.sqrt(bin_edges[:-1]) - shift))
        half_chi2_hist[0] += norm.cdf(-shift)  # Add the delta function at zero.
        half_chi2_hist[-1] += norm.sf(np.sqrt(upper_bound) - shift) # Add the tail above the upper bound to the last bin.
        half_chi2_hist /= bin_width  # Normalize by the bin width to get the probability density.
        plt.stairs(half_chi2_hist, bin_edges, fill = False, label='b, asymp.', color='red')

        # Plot various test statistic values.
        # -----------------------------------

        # self.ts and self.non_centrality as vertical lines.
        plt.axvline(self.ts, color='black', linestyle='-', label='Obs. t.s.')
        plt.axvline(self.non_centrality, color='cyan', linestyle='dashed', label='Asimov t.s.')

        # Quantiles corresponding to the n_sigma standard deviations for the expected limits, as vertical lines.
        for n_sigma, color in zip([-2, -1, 0, 1, 2], ['yellow', 'green', 'black', 'green', 'yellow']):
            probability = scipy.stats.norm.cdf(n_sigma)
            quantile = np.percentile(self.dict_pseudoexperiments[(self.mu_tested, 0)], probability * 100)
            plt.axvline(quantile, color=color, linestyle='dashed', label=f'Exp. t.s. {n_sigma}σ')

        plt.xlabel('Test statistic value')
        plt.ylabel('Probability density')
        plt.yscale('log')
        plt.title('Distribution of Test Statistic from Pseudo-experiments')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.outdir}/pseudoexperiments_{self.mu_tested:.2f}_{0:.2f}.png')
        plt.close()

        # Evaluate the chi-square goodness of fit test for the histogram of the test statistic values from the pseudo-experiments, comparing it to the half chi-square distribution with 1 degree of freedom.
        # For this, evaluate the expected number of pseudo-experiments in each bin according to the half chi-square distribution, and compare it to the observed number of pseudo-experiments in each bin.
        observed_counts = h * self.n_pseudoexperiments * bin_width
        expected_counts = half_chi2_hist * self.n_pseudoexperiments * bin_width
        chi_squared_value = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
        print(f'Chi-squared value: {chi_squared_value}')
        # Calculate the p-value for the chi-square test with the appropriate number of degrees of freedom, which is the number of bins.
        p_value = chi2.sf(chi_squared_value, df=n_bins)
        print(f'Chi-squared test p-value: {p_value}')

        return

    def histogram_pseudoexperiments(self, pseudoexperiments, bin_edges):
        """
        Histogram the test statistic values from the pseudo-experiments and calculate the uncertainty on the histogram counts.

        Parameters:
        -----------
        pseudoexperiments: array-like
            The test statistic values from the pseudo-experiments.
        bin_edges: array-like
            The edges of the bins for the histogram.

        Returns:
        --------
        h, h_unc: ndarray
            The histogram of the test statistic values from the pseudo-experiments, normalized to form a probability density.
            The uncertainty on the histogram counts, normalized in the same way as the histogram.
        """
        h, _ = np.histogram(pseudoexperiments, bins = bin_edges)
        h[0] += np.sum(pseudoexperiments < 0)

        # Add the events with test statistic value above the upper_bound to the last bin.
        h[-1] += np.sum(pseudoexperiments > bin_edges[-1])

        # Uncertainty on the histogram counts.
        h_unc = np.sqrt(h)

        # Normalize the histogram to form a probability density.
        bin_width = bin_edges[1] - bin_edges[0]
        h     = h / (self.n_pseudoexperiments * bin_width)
        h_unc = h_unc / (self.n_pseudoexperiments * bin_width)

        return h, h_unc


    
if __name__ == "__main__":
    # s = np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 2, 4, 6, 8, 10, 12, 14, 2, 4, 6, 8, 10, 12, 14]) * 0.5
    s = np.array([1.006524, 0.948773, 0.9858989, 1.056026, 1.072526, 1.167403, 1.485036, 1.592289, 1.612914, 1.678916, 2.165677, 2.330681, 2.211054, 2.145052, 2.025424, 2.392558, 2.149177, 1.827419, 1.905796, 1.563413, 1.518037, 2.037799, 1.151231, 1.056284, 1.145297, 1.240244, 1.258046, 1.418269, 1.620031, 2.160042, 2.225318, 2.302462, 2.990827, 3.251931, 3.0383, 3.032366, 2.866209, 3.412154, 2.907748, 2.516092, 2.676315, 2.290593, 2.189712, 2.854341])
    limits = Limits(s = s)
    # limits.set_cms_inputs("data/cms_monov_inputs.pkl")
    limits.set_cms_inputs("data/cms_monoj_inputs.pkl")
    # limits.pseudoexperiments(n_pseudoexperiments = 1000, read_pseudoexperiments_from_file = True, pseudoexperiments_file = "limits/dict_pseudoexperiments.pkl")
    results = limits.limits(np.linspace(2, 8, 4))

    # Just for the code-development phase: pickle the self.dict_pseudoexperiments to a file.
    with open("limits/dict_pseudoexperiments.pkl", "wb") as f:
        pickle.dump(limits.dict_pseudoexperiments, f)

    # Read in the results from the pickle file and print the observed and expected limits.
    with open("limits/results.pkl", "rb") as f:
        results = pickle.load(f)
    print(f'Observed upper limit on mu at 95% CL: {results["obs_limit"]}')
    for n_sigma in results["exp_limits"]:
        print(f'Expected upper limit on mu at 95% CL for n_sigma = {n_sigma}: {results["exp_limits"][n_sigma]}')
