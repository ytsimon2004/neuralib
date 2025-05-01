Spikes
=========

OASIS
----------------

Fast online deconvolution of calcium imaging dat

- **Refer to API**: :doc:`../api/neuralib.imaging.spikes.oasis`


.. seealso::

    - `Source Github <https://github.com/j-friedrich/OASIS>`_

    - `Friedrich et al., 2017. PLOS Computational Biology <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423>`_


**Example of usage**

.. code-block:: python

    from neuralib.imaging.spikes.oasis import oasis_dcnv

    # 2D dF/F array. Array[float, [nNeurons, nFrames]] or Array[float, nFrames]
    dff = ...

    tau = 1.5  # time constant of the calcium indicator (ms)
    fs = 30  # sampling frequency of the calcium imaging data (hz)
    spks = oasis_dcnv(dff, tau, fs)


Cascade
----------------

Wrapper class from Cascade to translate calcium imaging ΔF/F traces into spiking probabilities or discrete spikes

- **Refer to API**: :doc:`../api/neuralib.imaging.spikes.cascade`

- ``tensorflow`` required

- See available model in :meth:`~neuralib.imaging.spikes.cascade.CascadeSpikePrediction.get_available_models()`

.. seealso::

    - `Source Github <https://github.com/HelmchenLabSoftware/Cascade>`_

    - `Source notebook <https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb#scrollTo=cObwxWaB8i3f>`_


**Example of usage**


.. code-block:: python

    from neuralib.imaging.spikes.cascade import cascade_predict

    # 2D dF/F array. Array[float, [nNeurons, nFrames]] or Array[float, nFrames]
    dff = ...

    # select your model, predict the spike probability from the dF/F (same shape)
    spks = cascade_predict(dff, model_type='Global_EXC_30Hz_smoothing100ms')