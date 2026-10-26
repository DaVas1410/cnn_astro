.. docs/turbulens/method/index.rst

Method
======

How the training data were made, what the network looks like, how it is
trained, and what its uncertainty does and does not mean. Each page
ends with an "In turbulens" section naming the functions and
configuration keys that implement what it describes.

- :doc:`synthetic_data`: parameter ranges, splits and file layout of
  the synthetic training set, and where to get it.
- :doc:`preprocessing`: the exact pixel and output transformations,
  including the special handling of real observations.
- :doc:`architecture`: the shared ResNet backbone with one head per
  target.
- :doc:`training`: loss, optimizer, schedule and augmentation.
- :doc:`uncertainty`: deep ensembles and why their spread is epistemic
  only and not calibrated.
- :doc:`spectral_estimators`: the two classical, model-free estimators
  of the band edges.
- :doc:`evaluation`: definitions of the metrics the training pipeline
  reports.

.. toctree::
   :maxdepth: 1
   :hidden:

   synthetic_data
   preprocessing
   architecture
   training
   uncertainty
   spectral_estimators
   evaluation
