"""
Cascade
=========

**Cascade translates calcium imaging ΔF/F traces into spiking probabilities or discrete spikes**


See also the source example from the authors
----------------------------------------------

- `Github page <https://github.com/HelmchenLabSoftware/Cascade>`_

- `Calibrated spike inference with Cascade.ipynb <https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb#scrollTo=cObwxWaB8i3f>`_

- Since the source Cascade project not yet published in pypi, neuralib provide **wrapper usage** for Cascade(v1.0)


**Example of usage**

- See available model in :meth:`~neuralib.imaging.spikes.cascade.CascadeSpikePrediction.get_available_models()`

.. code-block:: python

    from neuralib.imaging.spikes.cascade import cascade_predict

    # 2D dF/F array. Array[float, [nNeurons, nFrames]] or Array[float, nFrames]
    dff = ...

    # select your model, predict the spike probability from the dF/F (same shape)
    spks = cascade_predict(dff, model_type='Global_EXC_30Hz_smoothing100ms')

"""
import re
import zipfile
from pathlib import Path
from typing import Literal, get_args, TypedDict
from urllib.request import urlopen

import numpy as np
import requests
import tensorflow as tf
import tensorflow.keras
from neuralib.io import CASCADE_MODEL_CACHE_DIRECTORY
from neuralib.typing import PathLike
from neuralib.util.utils import ensure_dir
from neuralib.util.verbose import fprint
from ruamel.yaml import YAML
from scipy.ndimage import binary_dilation, gaussian_filter

__all__ = [
    'CASCADE_MODEL_TYPE',
    'CascadeModelConfig',
    'cascade_predict',
    'CascadeSpikePrediction'
]

CASCADE_MODEL_TYPE = Literal[
    'Global_EXC_1Hz_smoothing500ms',
    'Global_EXC_1Hz_smoothing1000ms',
    'Zebrafish_1Hz_smoothing1000ms',
    'Global_EXC_2Hz_smoothing300ms',
    'Global_EXC_2Hz_smoothing500ms',
    'Global_EXC_2Hz_smoothing1000ms',
    'Global_EXC_2.5Hz_smoothing400ms_high_noise',
    'Global_EXC_3Hz_smoothing400ms',
    'Global_EXC_3Hz_smoothing400ms_high_noise',
    'Global_EXC_3Hz_smoothing400ms_causalkernel',
    'Global_EXC_4.25Hz_smoothing300ms',
    'Global_EXC_4.25Hz_smoothing300ms_high_noise',
    'Global_EXC_4.25Hz_smoothing300ms_causalkernel',
    'Global_EXC_5Hz_smoothing200ms',
    'Global_EXC_5Hz_smoothing200ms_causalkernel',
    'Global_EXC_6Hz_smoothing200ms',
    'Global_EXC_6Hz_smoothing200ms_causalkernel',
    'Global_EXC_7Hz_smoothing200ms',
    'Global_EXC_7Hz_smoothing200ms_causalkernel',
    'Global_EXC_7.5Hz_smoothing200ms_high_noise',
    'Global_EXC_7.5Hz_smoothing200ms',
    'Global_EXC_7.5Hz_smoothing200ms_causalkernel',
    'OGB_zf_pDp_7.5Hz_smoothing200ms',
    'OGB_zf_pDp_7.5Hz_smoothing200ms_causalkernel',
    'Global_EXC_10Hz_smoothing50ms',
    'Global_EXC_10Hz_smoothing50ms_causalkernel',
    'Global_EXC_10Hz_smoothing100ms',
    'Global_EXC_10Hz_smoothing100ms_causalkernel',
    'Global_EXC_10Hz_smoothing200ms',
    'Global_EXC_10Hz_smoothing200ms_causalkernel',
    'Global_EXC_12.5Hz_smoothing100ms',
    'Global_EXC_12.5Hz_smoothing100ms_causalkernel',
    'Global_EXC_12.5Hz_smoothing200ms',
    'Global_EXC_12.5Hz_smoothing200ms_causalkernel',
    'Global_EXC_15Hz_smoothing50ms',
    'Global_EXC_15Hz_smoothing50ms_causalkernel',
    'Global_EXC_15Hz_smoothing100ms_high_noise',
    'Global_EXC_15Hz_smoothing100ms',
    'Global_EXC_15Hz_smoothing100ms_causalkernel',
    'Global_EXC_15Hz_smoothing200ms',
    'Global_EXC_15Hz_smoothing200ms_causalkernel',
    'Global_INH_15Hz_smoothing100ms',
    'Global_EXC_17.5Hz_smoothing100ms',
    'Global_EXC_17.5Hz_smoothing100ms_causalkernel',
    'Global_EXC_17.5Hz_smoothing200ms',
    'Global_EXC_17.5Hz_smoothing200ms_causalkernel',
    'Global_EXC_20Hz_smoothing100ms',
    'Global_EXC_20Hz_smoothing100ms_causalkernel',
    'Global_EXC_20Hz_smoothing200ms',
    'Global_EXC_20Hz_smoothing200ms_causalkernel',
    'Global_EXC_25Hz_smoothing100ms',
    'Global_EXC_25Hz_smoothing100ms_causalkernel',
    'Global_EXC_25Hz_smoothing50ms',
    'Global_EXC_25Hz_smoothing50ms_causalkernel',
    'Global_EXC_30Hz_smoothing25ms',
    'Global_EXC_30Hz_smoothing25ms_causalkernel',
    'Global_EXC_30Hz_smoothing50ms',
    'Global_EXC_30Hz_smoothing50ms_high_noise',
    'Global_EXC_30Hz_smoothing50ms_causalkernel',
    'Global_EXC_30Hz_smoothing100ms',
    'Global_EXC_30Hz_smoothing100ms_causalkernel',
    'Global_EXC_30Hz_smoothing200ms',
    'Global_EXC_30Hz_smoothing100ms_causalkernel_high_noise',
    'Global_EXC_30Hz_smoothing100ms_high_noise',
    'Global_EXC_30Hz_smoothing200ms_causalkernel_high_noise',
    'Global_EXC_40Hz_smoothing25ms_causalkernel',
    'Global_EXC_40Hz_smoothing25ms',
    'Global_EXC_40Hz_smoothing25ms_high_noise',
    'Global_EXC_40Hz_smoothing50ms',
    'Global_EXC_40Hz_smoothing50ms_high_noise',
    'Global_EXC_40Hz_smoothing50ms_causalkernel',
    'Global_INH_30Hz_smoothing50ms',
    'Global_INH_30Hz_smoothing100ms',
    'Global_EXC_30Hz_smoothing50ms_asymmetric_window_1_frame',
    'Global_EXC_30Hz_smoothing50ms_asymmetric_window_2_frames',
    'Global_EXC_30Hz_smoothing50ms_asymmetric_window_4_frames',
    'Global_EXC_30Hz_smoothing50ms_asymmetric_window_6_frames',
    'Global_EXC_30Hz_smoothing50ms_asymmetric_window_8_frames',
    'GCaMP6f_mouse_30Hz_smoothing200ms',
    'Spinal_cord_excitatory_30Hz_smoothing50ms',
    'Spinal_cord_inhibitory_30Hz_smoothing50ms',
    'Spinal_cord_excitatory_3Hz_smoothing400ms_high_noise',
    'Spinal_cord_inhibitory_3Hz_smoothing400ms_high_noise',
    'Spinal_cord_excitatory_2.5Hz_smoothing400ms',
    'Spinal_cord_inhibitory_2.5Hz_smoothing400ms',
    'GC8_EXC_5Hz_smoothing400ms_high_noise',
    'GC8_EXC_5Hz_smoothing800ms_high_noise',
    'GC8_EXC_7.5Hz_smoothing100ms_high_noise',
    'GC8_EXC_7.5Hz_smoothing200ms_high_noise',
    'GC8_EXC_10Hz_smoothing150ms_high_noise',
    'GC8_EXC_10Hz_smoothing75ms_high_noise',
    'GC8_EXC_15Hz_smoothing100ms_high_noise',
    'GC8_EXC_15Hz_smoothing50ms_high_noise',
    'GC8_EXC_30Hz_smoothing25ms_high_noise',
    'GC8_EXC_30Hz_smoothing50ms_high_noise',
    'GC8_EXC_40Hz_smoothing15ms_high_noise',
    'GC8_EXC_40Hz_smoothing30ms_high_noise'
]


def cascade_predict(dff: np.ndarray,
                    model_type: CASCADE_MODEL_TYPE,
                    *,
                    threshold: int | bool = 0,
                    padding: float = 0,
                    verbose: bool = True,
                    chunks_mode_limit: float = 10,
                    cache_dir: PathLike | None = None) -> np.ndarray:
    """
    Spike prediction using Cascade pretrained model

    :param dff: dF/F activity to be predicted. `Array[float, [N, F]|F]`
    :param model_type: ``MODEL_TYPE``
    :param threshold: Allowed values: 0, 1 or False.
        0: All negative values are set to 0.
        1 or True: Threshold signal to set every signal which is smaller than the expected signal size of an action potential to zero (with dilated mask)
        False: No thresholding. The result can contain negative values as well
    :param padding: Value which is inserted for datapoints, where no prediction can be made
        (because of window around timepoint of prediction).
        Default value: np.nan, another recommended value would be 0 which circumvents some problems with following analysis.
    :param verbose: Verbose of model information
    :param chunks_mode_limit: Decrease the number if memory issue dealing with large input arrays.
    :param cache_dir: Cache directory for saving the model. If None, then used default under `~/.cache/neuralib`
    :return: Spiking probability as predicted by the model. `Array[float, [N, F]|F]`
    """
    cascade = CascadeSpikePrediction(
        dff,
        model_type,
        threshold=threshold,
        padding=padding,
        verbose=verbose,
        chunks_mode_limit=chunks_mode_limit,
        cache_dir=cache_dir
    )

    spike = cascade.run_spike_prediction()
    return spike


class CascadeModelConfig(TypedDict, total=False):
    model_name: str
    """Name of the model"""

    sampling_rate: int
    """Sampling rate in Hz"""

    training_datasets: list[str]
    """Dataset of ground truth data (in folder 'Ground_truth')"""

    placeholder_1: int
    """protect formatting"""

    noise_levels: list[int]
    """Noise levels for training (integers, normally 1-9)"""

    placeholder_2: int
    """protect formatting"""

    smoothing: float
    """Standard deviation of Gaussian smoothing in time (sec)"""

    causal_kernel: int
    """Smoothing kernel is symmetric in time (0) or is causal (1)"""

    windowsize: int
    """Windowsize in timepoints"""

    before_frac: float
    """Fraction of timepoints before prediction point (0-1)"""

    filter_sizes: list[int]
    """Filter sizes for each convolutional layer"""

    filter_numbers: list[int]
    """Filter numbers for each convolutional layer"""

    dense_expansion: int
    """For dense layer"""

    loss_function: str
    """gradient-descent loss function"""

    optimizer: str
    """Adagrad"""

    nr_of_epochs: int
    """Number of training epochs per model"""

    ensemble_size: int
    """Number of models trained for one noise level"""

    batch_size: int
    """Batch size"""

    training_finished: Literal['Yes', 'No', 'Running']
    """Yes / No / Running"""

    verbose: int
    """level of status messages (0: minimal, 1: standard, 2: most, 3: all)"""


class CascadeSpikePrediction:

    def __init__(
            self,
            dff: np.ndarray,
            model_type: CASCADE_MODEL_TYPE,
            *,
            threshold: int | bool = 0,
            padding: float = 0,
            verbose: bool = True,
            chunks_mode_limit: float = 10,
            cache_dir: PathLike | None = None
    ):
        """

        :param dff: dF/F activity to be predicted. `Array[float, [N, F]|F]`
        :param model_type: ``MODEL_TYPE``
        :param threshold: Allowed values: 0, 1 or False.
                0: All negative values are set to 0.
                1 or True: Threshold signal to set every signal which is smaller than the expected signal size of an action potential to zero (with dilated mask)
                False: No thresholding. The result can contain negative values as well
        :param padding: Value which is inserted for datapoints, where no prediction can be made (because of window around timepoint of prediction).
            Default value: np.nan, another recommended value would be 0 which circumvents some problems with following analysis.
        :param verbose: Verbose of model information
        :param chunks_mode_limit: Decrease the number if memory issue dealing with large input arrays.
        :param cache_dir: Cache directory for saving the model. If None, then used default under `~/.cache/neuralib`
        """
        self.dff = dff
        self.model_type = model_type

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir or CASCADE_MODEL_CACHE_DIRECTORY

        # model io
        if not self.available_model_yaml.exists():
            self._download_model_yaml()
        self._check_instance()
        if not self.model_dir.exists():
            self._download_model()

        # predict instance
        self.threshold = threshold
        self.padding = padding
        self.verbose = verbose
        self.chunks_mode_limit = chunks_mode_limit

    def _download_model_yaml(self):
        url = 'https://raw.githubusercontent.com/HelmchenLabSoftware/Cascade/master/Pretrained_models/available_models.yaml'
        response = requests.get(url)

        if response.status_code == 200:
            ensure_dir(self.cache_dir)
            out = self.available_model_yaml
            with open(out, 'wb') as file:
                file.write(response.content)
                fprint(f'YAML file saved successfully as {out}', vtype='io')
        else:
            raise RuntimeError(f'Failed to download the file. Status code: {response.status_code}')

    def _check_instance(self):
        available_models = self.get_available_models()
        if self.model_type not in available_models:
            raise ValueError(f'{self.model_type} should be one of the {available_models}')

        # check update
        if set(available_models) != set(get_args(CASCADE_MODEL_TYPE)):
            fprint('update MODEL_TYPE, new model released', vtype='warning')

    def _download_model(self):
        link = self.model_link
        with urlopen(link) as response:
            data = response.read()

        tmp_file = self.cache_dir / 'tmp_zipped_model.zip'
        with open(tmp_file, 'wb') as f:
            f.write(data)

        with zipfile.ZipFile(tmp_file, "r") as zip_ref:
            zip_ref.extractall(path=self.model_dir)

        tmp_file.unlink()

        fprint(f'Pretrained model was saved in folder {self.model_dir}', vtype='io')

    # ============== #
    # All Model Info #
    # ============== #

    @property
    def available_model_yaml(self) -> Path:
        """All models link/info yaml"""
        return self.cache_dir / 'available_models.yaml'

    def get_available_models(self) -> list[CASCADE_MODEL_TYPE]:
        """Get all the available in :attr:`available_model_yaml`"""
        content = YAML().load(self.available_model_yaml)
        models = list(content.keys())
        return models

    # ============== #
    # A Model Config #
    # ============== #

    @property
    def model_link(self) -> str:
        """Link of specified model"""
        with open(self.available_model_yaml, 'r') as file:
            config = YAML().load(file)
        return config[self.model_type]['Link']

    @property
    def model_dir(self) -> Path:
        """Directory of specified model"""
        return self.cache_dir / self.model_type

    @property
    def config_file(self):
        """Config filepath of specified model"""
        return self.model_dir / 'config.yaml'

    def get_config(self) -> CascadeModelConfig:
        """``ModelConfig`` of specified model"""
        with open(self.config_file, 'r') as file:
            return YAML().load(file)

    # ================ #
    # Spike Prediction #
    # ================ #

    def run_spike_prediction(self) -> np.ndarray:
        """
        Spike prediction

        :return: Spiking probability as predicted by the model. `Array[float, [N, F]|F]`
        """
        total_array_size = self.dff.itemsize * self.dff.size * 64 / 1e9

        if total_array_size < self.chunks_mode_limit:
            spike = self._predict(self.dff, self.threshold, self.padding, self.verbose)
        else:
            n_neurons = self.dff.shape[0]
            n_frames = self.dff.shape[1]
            spike = np.zeros((n_neurons, n_frames))
            nb_chunks = int(np.ceil(total_array_size / 10))
            chunks = np.array_split(range(n_neurons), nb_chunks)
            for part in range(nb_chunks):
                part_dff = self.dff[chunks[part], :]
                spike[chunks[part], :] = self._predict(part_dff, self.threshold, self.padding, self.verbose)

        return spike

    def _predict(self, dff, threshold=0, padding: float = np.nan, verbose: bool = True):
        # reshape if only a single neuron's activity is provided
        if len(dff.shape) == 1:
            dff = np.expand_dims(dff, 0)

        cfg = self.get_config()
        verbose = cfg["verbose"]
        training_data = cfg["training_datasets"]
        ensemble_size = cfg["ensemble_size"]
        batch_size = cfg["batch_size"]
        sampling_rate = cfg["sampling_rate"]
        before_frac = cfg["before_frac"]
        window_size = cfg["windowsize"]
        noise_levels_model = cfg["noise_levels"]
        smoothing = cfg["smoothing"]
        causal_kernel = cfg["causal_kernel"]

        # calculate noise levels for each trace
        trace_noise_levels = calculate_noise_levels(dff, sampling_rate)

        # Get model paths as dictionary (key: noise_level) with lists of model path
        model_dict = get_model_paths(self.model_dir)

        if verbose:
            msg = (f'The selected model was trained on {len(training_data)} datasets, '
                   f'with {ensemble_size} ensembles for each noise level, at a sampling rate of {sampling_rate} Hz')

            if causal_kernel:
                msg += ", with a resampled ground truth that was smoothed with a causal kernel"
            else:
                msg += ", with a resampled ground truth that was smoothed with a Gaussian kernel"

            msg += f'of a standard deviation of {str(int(1000 * smoothing))} ms.'
            fprint(msg)

            fprint(f'Loaded model was trained at frame rate {sampling_rate} Hz')
            fprint(f'Given argument traces contains {dff.shape[0]} neurons and {dff.shape[1]} frames.')

            noise_mean = str(int(np.nanmean(trace_noise_levels * 100)) / 100)
            noise_std = str(int(np.nanstd(trace_noise_levels * 100)) / 100)
            fprint(f'Noise levels (mean, std; in standard units): {noise_mean}, {noise_std}')

        # XX has shape: (neurons, timepoints, windows)
        XX = preprocess_traces(dff, before_frac=before_frac, window_size=window_size)
        Y_predict = np.zeros((XX.shape[0], XX.shape[1]))

        # Compute difference of noise levels between each neuron and each model; find the best fit
        differences = np.array(trace_noise_levels)[:, None] - np.array(noise_levels_model)[None, :]
        relative_differences = np.min(differences, axis=1)
        if np.mean(relative_differences) > 2:
            fprint(
                f"WARNING: The available models cannot match the experimentally obtained noise levels (difference: {str(np.mean(relative_differences))}),"
                f"Please check that the computation of dF/F is performed correctly. Otherwise, please reach out and ask for pretrained models with higher noise level models "
                f"(see: https://github.com/HelmchenLabSoftware/Cascade/issues/61).", vtype='warning'
            )
        best_model_for_each_neuron = np.argmin(np.abs(differences), axis=1)

        # Use for each noise level the matching model
        for i, noise_level in enumerate(noise_levels_model):

            # select neurons which have this noise level:
            neuron_idx = np.where(best_model_for_each_neuron == i)[0]
            if verbose:
                print(f'\nPredictions for noise level {noise_level}')
            if len(neuron_idx) == 0:  # no neurons were selected
                if verbose:
                    print(f"\tNo neurons for this noise level: {noise_level}")
                continue  # jump to next noise level

            # load keras models for the given noise level
            models = []
            for model_path in model_dict[noise_level]:
                models.append(tf.keras.models.load_model(model_path))

            # select neurons and merge neurons and timepoints into one dimension
            XX_sel = XX[neuron_idx, :, :]

            XX_sel = np.reshape(XX_sel, (XX_sel.shape[0] * XX_sel.shape[1], XX_sel.shape[2]))
            XX_sel = np.expand_dims(XX_sel, axis=2)  # add empty third dimension to match training shape

            for j, model in enumerate(models):
                if verbose:
                    print("\t... ensemble", j)

                prediction_flat = model.predict(XX_sel, batch_size, verbose=verbose)
                prediction = np.reshape(prediction_flat, (len(neuron_idx), XX.shape[1]))
                Y_predict[neuron_idx, :] += prediction / len(models)  # average predictions

            # remove models from memory
            tensorflow.keras.backend.clear_session()

        # handle threshold
        if threshold is False:
            fprint('Skipping the thresholding. There can be negative values in the result.', vtype='warning')

        elif threshold == 1:  # or True
            # Cut off noise floor (lower than 1/e of a single action potential)
            # find out empirically  how large a single AP is (depends on frame rate and smoothing)
            single_spike = np.zeros(1001)
            single_spike[501] = 1
            single_spike_smoothed = gaussian_filter(single_spike.astype(float), sigma=smoothing * sampling_rate)
            threshold_value = np.max(single_spike_smoothed) / np.exp(1)

            # Set everything below threshold to zero.
            # Use binary dilation to avoid clipping of true events.
            for neuron in range(Y_predict.shape[0]):
                # ignore warning because of nan's in Y_predict in comparison with value
                with np.errstate(invalid="ignore"):
                    activity_mask = Y_predict[neuron, :] > threshold_value

                activity_mask = binary_dilation(activity_mask, iterations=int(smoothing * sampling_rate))
                Y_predict[neuron, ~activity_mask] = 0
                Y_predict[Y_predict < 0] = 0  # set possible negative values in dilated mask to 0

        elif threshold == 0:
            # ignore warning because of nan's in Y_predict in comparison with value
            with np.errstate(invalid="ignore"):
                Y_predict[Y_predict < 0] = 0

        else:
            raise ValueError(f'Invalid value of threshold "{threshold}". Only 0, 1 (or True) or False allowed')

        # NaN or 0 for first and last datapoints, for which no predictions can be made
        Y_predict[:, 0: int(before_frac * window_size)] = padding
        Y_predict[:, -int((1 - before_frac) * window_size):] = padding

        return Y_predict


def preprocess_traces(dff: np.ndarray,
                      before_frac: float,
                      window_size: int) -> np.ndarray:
    """
    Transform dF/F data into a format that can be used by the deep network.
    For each time point, a window of the size 'window_size' of the dF/F is extracted.

    :param dff: `Array[float, [N, F]]`
    :param before_frac: positioning of the window around the current time point; 0.5 means center position
    :param window_size: size of the receptive window of the deep network
    :return: a matrix with `Array[float, [N, F, W]]`

    """
    start = int(before_frac * window_size - 1)
    end = dff.shape[1] - window_size + start + 1

    # extract a moving window from the calcium trace
    window_indexes = (
            np.expand_dims(np.arange(window_size), 0) +
            np.expand_dims(np.arange(dff.shape[1] - window_size + 1), 0).T
    )

    X = np.full((*dff.shape, window_size), np.nan)
    X[:, start:end, :] = dff[:, window_indexes]

    return X


def get_model_paths(model_path: Path) -> dict[int, list[Path]]:
    """Find all models in the model folder and return as dictionary"""
    all_models = sorted(list(model_path.glob('*.h5')))
    if len(all_models) == 0:
        raise FileNotFoundError(f'No models (*.h5 files) were found in the specified folder "{model_path}".')

    # dictionary with key for noise level, entries are lists of models
    model_dict = {}
    for model_path in all_models:
        noise_level = int(re.findall("_NoiseLevel_(\d+)", str(model_path))[0])
        if noise_level not in model_dict:
            model_dict[noise_level] = list()

        model_dict[noise_level].append(model_path)

    return model_dict


def calculate_noise_levels(dff: np.ndarray, frame_rate: float) -> np.ndarray:
    """
    Computes the noise levels for each neuron of the input matrix 'dF_traces'.

    The noise level is computed as the median absolute dF/F difference
    between two subsequent time points. This is a outlier-robust measurement
    that converges to the simple standard deviation of the dF/F trace for
    uncorrelated and outlier-free dF/F traces.

    Afterwards, the value is divided by the square root of the frame rate
    in order to make it comparable across recordings with different frame rates.

    :param dff: Fluorescence changed dF/F. `Array[float, [N, F]]`
    :param frame_rate: frame rate
    :return: vector of noise levels for all neurons
    """
    noise_levels = np.nanmedian(np.abs(np.diff(dff, axis=-1)), axis=-1) / np.sqrt(frame_rate)
    return noise_levels * 100  # scale noise levels to percent
