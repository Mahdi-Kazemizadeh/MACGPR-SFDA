# MACGPR

MLLM-Assisted Cluster-Guided Pseudo-label Refinement for Source-Free Domain Adaptation.

## Project Goal

This repository implements a source-free domain adaptation pipeline for VisDA-C Synthetic to Real.

The project starts with a CGPR baseline and then extends it with selective MLLM-assisted pseudo-label refinement.

## Dataset

The VisDA-C dataset is not included in this repository.

Set the dataset root using:

```bash
VISDA_ROOT=/path/to/visda-c