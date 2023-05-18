import os
import subprocess
import sys
from cmdstanpy import cmdstan_path, CmdStanModel

# run pip check
subprocess.run(['pip', 'check'])

# Set correct path to SDK for tests
platform = sys.platform
if platform == "darwin":
    host = os.environ["HOST"]
    if "arm64" in host:
        os.environ["CONDA_BUILD_SYSROOT"] = '/Library/Developer/CommandLineTools/SDKs/MacOSX11.1.sdk'
    else:
        os.environ["CONDA_BUILD_SYSROOT"] = '/opt/MacOSX10.10.sdk'

# specify locations of Stan program file and data
bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')

# instantiate a model; compiles the Stan program by default
bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

# obtain a posterior sample from the model conditioned on the data
bernoulli_fit = bernoulli_model.sample(chains=4, data=bernoulli_data)

# summarize the results (wraps CmdStan `bin/stansummary`):
bernoulli_fit.summary()

# # test with threading
# # instantiate a model; compiles the Stan program by default
# bernoulli_model = CmdStanModel(stan_file=bernoulli_stan, compile='force', cpp_options={'STAN_THREADS':True})

# # obtain a posterior sample from the model conditioned on the data
# bernoulli_fit = bernoulli_model.sample(chains=4, data=bernoulli_data, parallel_chains=2)

# # summarize the results (wraps CmdStan `bin/stansummary`):
# bernoulli_fit.summary()
