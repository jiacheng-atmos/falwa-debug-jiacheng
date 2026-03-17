# FALWA NHN22 blow-up issue

This repository contains a minimal example demonstrating numerical instability (blow-up) in the NHN22 implementation of FALWA.

## Issue summary

- Blow-up occurs when static stability becomes very small
- The stretching term becomes excessively large
- The resulting reference state shows unrealistic alternating structures

## Contents

- `nhn22_blowup_analysis.ipynb`: main notebook reproducing the issue

## How to reproduce

1. Run the notebook
2. Check the evolution of:
   - stat_n
   - stat_s
   - stat_18
3. Blow-up occurs around timestep: XXX

## Notes

- NH18 implementation does not show this instability
- Difference in stretching term (zmav vs f) is not the main cause
