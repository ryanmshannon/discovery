# Daniel J. Reardon -- danieljohnreardon@gmail.com #

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import glob

#

#utils.setup_gpu_environment()


import jax
jax.config.update("jax_enable_x64", False)


import jax.random
import numpy as np
import pandas as pd
import discovery as ds
import discovery.samplers.numpyro as ds_numpyro
import discovery.models.mpta as mpta
import discovery.models.ppta as ppta
#import discovery.gpu_utils as utils
ds.gpu_utils.setup_gpu_environment()



import argparse
from pta_utils import write_ml_json


# Add arguments
parser = argparse.ArgumentParser(description="Discovery common noise/signal analysis script")
parser.add_argument("-dir", dest="dir", help="Directory of pulsar feather objects to use", required=True)
parser.add_argument("-chainnum", dest="chainnum", help="Chain number", default='1', required=False)
parser.add_argument("-noisefiles", dest="noisefiles", help="File containing names of single pulsar analysis data frames to use", required=True)
parser.add_argument("-outdir", dest="outdir", help="Directory to save results to", required=False, default=None)
parser.add_argument("-model", dest="model", help="Common noise model to use", required=False, default='curn', 
                    choices=['curn', 'gwb', 'curn_fixgamma', 'gwb_fixgamma', 'cgw', 'cgw+curn', 'cgw+gwb', 'ecgw', 'ecgw+curn', 'ecgw+gwb']) # Note: Choices are not implemented. This is a wish list

# Parse arguments
args = parser.parse_args()
feather_dir = args.dir
noisefiles = args.noisefiles
model_name = args.model
chain_number = args.chainnum
outdir = args.outdir if args.outdir is not None else feather_dir    

# Check how many GPUs are available
num_gpus = ds.gpu_utils.get_num_gpus()
print(f"Available GPUs: {num_gpus}")

# Get list of GPU devices
devices = ds.gpu_utils.get_gpu_devices()
print(f"Devices: {devices}")

# Load in pulsar feathers
psrs = []
for fp in sorted(glob.glob(feather_dir + "/*.feather")):
    psr = ds.Pulsar.read_feather(fp)
    psrs.append(psr)

# Read in single pulsar analysis chains
chain_dfs = []
with open(noisefiles, 'r') as f:
    # sort file names by pulsar name
    file_names = sorted([line.strip() for line in f], key=lambda x: os.path.basename(x).split('_')[0])
    for fn in file_names:
        print("Reading in chain:", fn)
        df = pd.read_pickle(fn)
        chain_dfs.append(df)

# Choose model. EFAC, EQUAD, and ECORR, timing model, are always included.
if "mpta" in feather_dir:
    max_cadence_days = 14
    mpta.update_priordict_standard_mpta()
elif "ppta" in feather_dir:
    max_cadence_days = 30
    ppta.update_priordict_standard_ppta()




def common_noise(psrs, chain_dfs, fftInt=True, max_cadence_days=14, name="gw_crn"):
    # Accepts a list of pulsars and their corresponding chain dataframes and constructs an ArrayLikelihood
    def has_param(df, param_string="red_noise"):
        return any(f"{param_string}" in col for col in list(df.columns))

    Tspan = ds.signals.getspan(psrs)
    common_components = int(Tspan / (max_cadence_days * 86400))
    common_knots = 2 * common_components + 1

    psls = []
    for psr, df in zip(psrs, chain_dfs):
        if not any(psr.name in col for col in df.columns):
            raise ValueError("Chain data frames do not match pulsar names")
        # Get max-likelihood parameters for this pulsar
        ml_idx = df['logl'].idxmax()
        noisedict = {col: df.loc[ml_idx, col] for col in df.columns if col.startswith(psr.name)}

        if not fftInt:
            curn = ds.signals.makegp_fourier(psr, ds.signals.powerlaw, common_components, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        else:
            curn = ds.signals.makegp_fftcov(psr, ds.signals.powerlaw, common_knots, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        extra_gps = curn if isinstance(curn, list) else [curn]


        # background = False, as we are including a common red noise process
        m = mpta.single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
                                red=has_param(df, "red_noise"), dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"), sw=has_param(df, "sw_gp"),
                                band=has_param(df, "band_gp"), band_low=has_param(df, "band_low_gp"), band_alpha=has_param(df, "bandalpha_gp"),
                                chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"),
                                extra_gps=extra_gps)
        
        #m = mpta.single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
        #                        red=has_param(df, "red_noise"), dm=has_param(df, "dm_gp"), extra_gps=extra_gps)


        print("Including pulsar", psr.name)
        psls.append(m)

    return ds.likelihood.GlobalLikelihood(psls)

model = common_noise(psrs, chain_dfs, fftInt=True, max_cadence_days=max_cadence_days, name=model_name)

logl_parallel = model.gpu_logL(use_pmap=True)


#print("Model parameters:\n", logl_parallel.params)

# Save results as a DataFrame
save_name = "{0}/results/{1}_{2}_{3}".format(outdir, model_name, str(max_cadence_days), str(chain_number))
os.makedirs("{0}/results/".format(outdir), exist_ok=True)
print("Saving results to", save_name)


# Set up sampler, sample, and save results
npmodel=  ds_numpyro.makemodel_transformed(logl_parallel)
sampler = ds_numpyro.makesampler_nuts(npmodel, num_warmup=512, num_samples=100)
# start with random seed
key = jax.random.PRNGKey( np.random.randint(0, 2**32 - 1) )
sampler.run(key)

chain = sampler.to_df()
chain.to_pickle(save_name + ".pickle")
write_ml_json(chain, save_name + ".json")
# Skip making plots because they are likely too large. Use "plot_corner.py" on the pickle file instead.
# sampler.make_plots(save_name=save_name)
