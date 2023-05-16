import os
import subprocess
from cmdstanpy import cmdstan_path, CmdStanModel

# run pip check
subprocess.run(['pip', 'check'])

#os.environ["CPPFLAGS"] = '-D_FORTIFY_SOURCE=2 -isystem $PREFIX/include -mmacosx-version-min=10.14'

#r=subprocess.run(['clang++ --version'],shell=True)
r = subprocess.Popen('clang++ --version', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read().decode('utf-8').splitlines())

r = subprocess.Popen('echo $CXX', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read())

r = subprocess.Popen('echo $CPPFLAGS', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read())

r = subprocess.Popen('echo $SDKROOT', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read())

r = subprocess.Popen('readlink -f $SDKROOT', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read())

r = subprocess.Popen('echo $CONDA_BUILD_SYSROOT', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read())

r = subprocess.Popen('readlink -f $CONDA_BUILD_SYSROOT', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read())

r = subprocess.Popen('echo $MACOSX_DEPLOYMENT_TARGET', shell=True, stdout=subprocess.PIPE)
print(r.stdout.read())


# specify locations of Stan program file and data
bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')

# instantiate a model; compiles the Stan program by default
bernoulli_model = CmdStanModel(stan_file=bernoulli_stan, cpp_options={'mmacosx-version-min':"10.14"})

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
