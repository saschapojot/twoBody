1. To calculate pbc eigenvalue problem for omegaF not equal to 0, use oneTorchSpectrum.py; to calculate pbc eigenvalue problem for omegaF=0, use oneTorchSpectrumOmegaF0.py. 
2. Using results of eigenvectors from step 1, to calculate Wannier state pumping, use readFromTab.py; to calculate Gaussian state pumping, use gaussianReadFromTab.py. The Chern number for the selected band is computed before pumping computation.
3. To compute obc eigenvalue problem, use torchObc.py
