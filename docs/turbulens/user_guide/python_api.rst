.. docs/turbulens/user_guide/python_api.rst

Using turbulens from Python
===========================

Everything the CLI does is available as a library. This example
reproduces the getting-started run.

.. code-block:: python

   from astropy import units as u

   from turbulens.inference import Inferencer
   from turbulens.io.image_loaders import load_input
   from turbulens.models.registry import EnsembleRegistry
   from turbulens.spectral import SpectralEstimator

   # Load a cube and integrate it over -100..100 km/s
   cube = load_input("turbulens/io/tests/data/gass_0_0_1774393774.fits.gz")
   image = cube.project_velocity(-100 * u.km / u.s, 100 * u.km / u.s).data[0]

   # Download (first time) or reuse the cached multitask ensemble
   fetched = EnsembleRegistry.fetch()
   table = Inferencer(fetched.members_dir).predict(image, real_data=True)
   print(table)
   print(fetched.repo_id, fetched.revision)

   # Model-free cross-check
   classical = SpectralEstimator(method="power_law").predict(image)
   print(classical)

The pieces
----------

`turbulens.io.image_loaders.load_input`
   Reads FITS, ``.npy``, PNG or TIFF into a `turbulens.io.cube.Cube`
   (:doc:`inputs`).

`turbulens.io.cube.Cube`
   Always ``(n_channels, H, W)``. ``slice_channel``, ``project`` (channel
   indices) and ``project_velocity`` (velocities) each return a
   one-channel cube; ``.data[0]`` is the 2D image. ``velocity_axis``
   gives the channel velocities.

`turbulens.models.registry.EnsembleRegistry`
   ``fetch(target=None, version="latest", repo_id=..., revision=None)``
   downloads one ensemble from the Hub and returns a ``FetchedEnsemble``
   with ``members_dir``, ``repo_id`` and ``revision``. ``resolve`` finds
   an ensemble under your own pipeline output root
   (:doc:`pretrained_models`).

`turbulens.inference.Inferencer`
   ``Inferencer(members_dir, device="cpu")`` loads all members.
   ``predict(images, real_data=False)`` takes one ``(H, W)`` image or an
   ``(N, H, W)`` batch and returns a table with ``image_index``,
   ``target``, ``value`` and ``epistemic_std``
   (:doc:`outputs_provenance`). Pass ``real_data=True`` for real
   observations (:ref:`real-data normalization <real-data-normalization>`).
   Use ``device="cuda"`` to run on a GPU.

`turbulens.spectral.SpectralEstimator`
   ``SpectralEstimator(method="band_edge" | "power_law").predict(images)``,
   the classical estimators behind the same table interface
   (:ref:`spectral-estimators`).

`turbulens.io.results.write_results`
   Writes a table to ``.ecsv``, ``.csv`` or ``.fits``.

`turbulens.io.breakdown.describe_cube`
   The summary behind ``turbulens inspect``, with text, JSON and HTML
   renderers in `turbulens.io.report`.

Batches
-------

``predict`` runs the whole batch through each member in one pass, so
memory grows with the batch size. Split very large batches into chunks
and concatenate the tables, offsetting ``image_index`` if you need a
global index.

The full reference for every class and function is in
:doc:`/api/index`.
