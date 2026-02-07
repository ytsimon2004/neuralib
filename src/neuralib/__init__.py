import warnings

warnings.warn(
    "The 'neura-library' package has been archived and is no longer maintained. "
    "Please migrate to the new modular packages from neuralib2: "
    "neuralib-atlas, neuralib-imaging, neuralib-parser, neuralib-metric, neuralib-utils. "
    "See https://github.com/ytsimon2004/neuralib2 for more information.",
    DeprecationWarning,
    stacklevel=2,
)
