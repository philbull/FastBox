#!/bin/bash
planck_maps="COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits COM_SimMap_freefree-ffp10-skyinbands-217_2048_R3.00_full.fits COM_SimMap_synchrotron-ffp10-skyinbands-353_2048_R3.00_full.fits"
echo "Downloading Planck Sky Model maps"
echo "(approx. 2.5 GB download)"
for f in $planck_maps; do
  echo "Downloading $f...";
  wget -O ../fastbox/$f "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=$f";
  done

echo "Downloading destriped Haslam map"
echo "(approx. 13 MB download)"
wget -O ../fastbox/haslam408_ds_Remazeilles2014.fits "https://lambda.gsfc.nasa.gov/product/foreground/fg_2014_haslam_408_get.html#:~:text=haslam408_ds_Remazeilles2014.fits"

echo "Downloading high resolution spectral index map"
echo "(approx. 24 MB download)"
wget -O ../fastbox/cnn56arcmin_beta.npy "https://github.com/melisirfan/synchrotron_emission/raw/main/cnn56arcmin_beta.npy"

echo "Finished."
