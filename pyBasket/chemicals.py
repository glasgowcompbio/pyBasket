from random import randrange

import numpy as np
from vimms.ChemicalSamplers import UniformMZFormulaSampler, UniformRTAndIntensitySampler, GaussianChromatogramSampler
from vimms.Chemicals import ChemicalMixtureCreator


def generate_chemicals(chemical_creator_params):
    """
    Generates new set of chemicals
    """
    n_chemicals_range = chemical_creator_params['n_chemicals']
    if n_chemicals_range[0] == n_chemicals_range[1]:
        n_chems = n_chemicals_range[0]
    else:
        n_chems = randrange(n_chemicals_range[0], n_chemicals_range[1])

    # sample m/z according to the sampler provided
    if 'mz_sampler' in chemical_creator_params:
        mz_sampler = chemical_creator_params['mz_sampler']
    else:
        # sample chemicals uniformly
        min_mz = chemical_creator_params['mz_range'][0]
        max_mz = chemical_creator_params['mz_range'][1]
        mz_sampler = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)

    # sample RT and intensity according to the sampler provided
    if 'ri_sampler' in chemical_creator_params:
        ri_sampler = chemical_creator_params['ri_sampler']
    else:
        # sample RT and intensity uniformly
        min_rt = chemical_creator_params['rt_range'][0]
        max_rt = chemical_creator_params['rt_range'][1]
        min_log_intensity = np.log(chemical_creator_params['intensity_range'][0])
        max_log_intensity = np.log(chemical_creator_params['intensity_range'][1])
        ri_sampler = UniformRTAndIntensitySampler(min_rt=min_rt, max_rt=max_rt,
                                                  min_log_intensity=min_log_intensity,
                                                  max_log_intensity=max_log_intensity)

    # sample chromatogram shapes according to the sampler provided
    if 'cr_sampler' in chemical_creator_params:
        cr_sampler = chemical_creator_params['cr_sampler']
    else:
        # generate default gaussian shaped chromatograms
        cr_sampler = GaussianChromatogramSampler()

    # put everything together
    cm = ChemicalMixtureCreator(mz_sampler, rt_and_intensity_sampler=ri_sampler,
                                chromatogram_sampler=cr_sampler)
    chems = cm.sample(n_chems, 2, include_adducts_isotopes=False)
    return chems


