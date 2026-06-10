# covli

Package for statistical analysis using different types of likelihood functions.

It uses the profile likelihood ratio test statistic and its asymptotic distributions described in [https://arxiv.org/abs/1007.1727](https://arxiv.org/abs/1007.1727)

# covli.py

Module for the limits determination using likelihood in the covariance matrix representation.

Description of the likelihood covariance representation: [https://arxiv.org/pdf/2307.04007](https://arxiv.org/pdf/2307.04007)

Note: implementation of the asymptotic formulae has just been fixed in covli.py, according to how they are implemented in simpli.py.
However, no tests were run, yet, to check that no new bugs were introduced.
Use this module with caution!

# simpli.py

Module for the limits determination using the CMS simplified likelihood.

Description of the simplified likelihood used by CMS: [https://cds.cern.ch/record/2242860](https://cds.cern.ch/record/2242860)

HEPData entry with the CMS monojet/V analysis results: [https://www.hepdata.net/record/ins1894408](https://www.hepdata.net/record/ins1894408)

Besides the asymptotic test statistic distributions, this module also allows to determine the distributions using pseudoexperiments.
