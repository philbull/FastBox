# FastBox
Fast simulations of cosmological density fields, 
subject to anisotropic filtering, biasing, 
redshift-space distortions, foregrounds etc. This 
is intended to be a fast and simple simulator for 
post-EoR 21cm intensity maps and their cross-correlation 
with galaxy samples, with enough complexity to test 
realistic cosmological analysis methods.

Current features include:
 - Gaussian and log-normal density fields for any cosmology
 - Redshift-space transform, linear biasing etc
 - Arbitrary anisotropic filters as a function of 
   k_perp and k_parallel
 - Poisson realisations of halo/galaxy samples
 - Radio noise and foregrounds
 - Simple foreground filtering (via PCA and ICA)
 - Integration with the DESC Core Cosmology Library
 - Power spectra, correlation functions, and their 
   multipoles, via nbodykit

