.. docs/turbulens/citing.rst

Citing turbulens
================

If you use `turbulens` in your work, please cite the software:

   Vasconez Vela, J. D., Banda-Barragán, W. E., & Morales Navarrete, H. A.
   *turbulens: inference tool for predicting fractal turbulence
   parameters from radio-astronomy FITS cubes.* Software, version |release|.

When you use the pretrained models or the synthetic training data, also
name the Hugging Face repositories and, for the models, the exact
revision recorded in your output table's ``model_revision``
(:doc:`user_guide/outputs_provenance`):

- models: ``Davas1410/turbulens-ensembles``
- training data: ``Davas1410/turbulens-synthetic-data``

Please also cite the methods and software `turbulens` builds on that
matter for your use, in particular pyFC :cite:p:`pyfc` and the
log-normal field construction of :cite:t:`lewis2002` for the training
data, deep ensembles :cite:p:`lakshminarayanan2016`, and residual
networks :cite:p:`he2016`. Full entries are in :doc:`references`.
