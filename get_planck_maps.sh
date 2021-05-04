#!/bin/bash
planck_maps="COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits COM_SimMap_freefree-ffp10-skyinbands-217_2048_R3.00_full.fits COM_SimMap_synchrotron-ffp10-skyinbands-353_2048_R3.00_full.fits"
echo "Downloading Planck Sky Model maps"
echo "(approx. 2.5 GB download)"
for f in $planck_maps; do
  echo "Downloading $f...";
  wget -O fastbox/$f "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=$f";
  done
echo "Finished."
