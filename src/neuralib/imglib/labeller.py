"""
SequenceLabeller
===================

Simple CV2-based viewer/labeller GUI for image sequences

Use Cases:

- viewing the image sequences

- label each image and save as csv dataframe (human-eval for population neurons activity profile)


Load sequences from a directory
-----------------------------------

- Use CLI mode

See help::

    python labeller.py -h

Example::

    python labeller.py -D <DIR>


- Use API call

.. code-block:: python

    from neuralib.imglib.labeller import SequenceLabeller

    directory = ...
    labeller = SequenceLabeller.load_from_dir(directory)
    labeller.main()


Load sequences from sequences array
-------------------------------------

.. code-block:: python

    from neuralib.imglib.labeller import SequenceLabeller

    arr = ...   # numpy array with (F, H, W, <3>)
    labeller = SequenceLabeller.load_sequences(arr)
    labeller.main()

"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import ClassVar, Callable, Any, Literal

import attrs
import cv2
import numpy as np
import polars as pl
from tifffile import tifffile
from tqdm import tqdm
from typing_extensions import Self

from neuralib.util.csv import csv_header
from neuralib.util.util_cv2 import get_keymapping, KeyMapping, find_key_from_value
from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint

__all__ = ['SequenceLabeller']

Logger = logging.getLogger()


@attrs.define
class FrameInfo:
    filename: str
    """name of the an image/frame"""
    image: np.ndarray
    """(H, W, <3>)"""
    notes: str | None = attrs.field(default=None)
    """notes for the image"""

    @property
    def itype(self) -> Literal['gray', 'rgb']:
        if self.image.ndim == 2:
            return 'gray'
        elif self.image.ndim == 3:
            return 'rgb'
        else:
            raise TypeError('')

    @property
    def text_color(self) -> float | tuple[int, int, int]:
        if self.itype == 'gray':
            return 2 ** 16 - 1
        elif self.itype == 'rgb':
            return 0, 0, 255
        else:
            raise RuntimeError('')

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


class CloseSaveInterrupt(KeyboardInterrupt):
    """write & quiet triggered"""
    pass


class SequenceLabeller:
    window_title: ClassVar[str] = 'SeqLabeller'

    def __init__(self, seqs_info: list[FrameInfo],
                 output: PathLike | None = None):
        self.seqs_info = seqs_info
        self.output = output  # for notes

        self.message_queue: list[str] = []
        self.buffer = ''  # input buffer
        self._frame_index = 0

    def __len__(self) -> int:
        return len(self.seqs_info)

    @classmethod
    def load_sequences(cls, seqs: np.ndarray | list[np.ndarray],
                       filenames: list[str] | None = None,
                       output: PathLike | None = None) -> Self:
        """

        :param seqs:
        :param filenames:
        :param output:
        :return:
        """
        if isinstance(seqs, np.ndarray):
            seqs = [it for it in seqs]

        n_frames = len(seqs)

        if filenames is None:
            filenames = np.arange(n_frames).astype(str)

        seqs_info = [FrameInfo(filenames[i], seqs[i], None) for i in range(n_frames)]

        return SequenceLabeller(seqs_info, Path(output) if output is not None else None)

    @classmethod
    def load_from_dir(cls, directory: PathLike,
                      file_suffix: str = '.tif',
                      sort_func: Callable[[str], Any] | None = None,
                      single_frame_per_file: bool = True,
                      output: PathLike | None = None) -> Self:
        """

        :param directory: directory contain image sequences
        :param file_suffix: sequence file suffix
        :param sort_func: sorted function
        :param single_frame_per_file:
        :param output:
        :return:
        """
        if not Path(directory).is_dir():
            raise NotADirectoryError(f'{directory}')

        files = sorted(list(directory.glob(f'*{file_suffix}')), key=sort_func)

        if len(files) == 0:
            raise FileNotFoundError('')
        else:
            fprint(f'LOAD image sequence: {len(files)} files', vtype='io')

        seqs = []
        for f in tqdm(files, unit='file', ncols=80):

            if file_suffix == '.pdf':
                from neuralib.imglib.io import read_pdf
                img = read_pdf(f, dpi=200)
                seqs.append(img)
            else:
                if single_frame_per_file:
                    img = cv2.imread(str(f))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    seqs.append(img)
                elif not single_frame_per_file and file_suffix in ('.tif', '.tiff'):
                    seqs.append(tifffile.imread(str(f)))
                else:
                    raise NotImplementedError('')

        filenames = [it.stem for it in files]
        seqs_info = [FrameInfo(filenames[i], seqs[i], None) for i in range(len(seqs))]

        return SequenceLabeller(seqs_info, Path(output) if output is not None else None)

    @property
    def n_frames(self) -> int:
        """aka. number of images"""
        return len(self.seqs_info)

    @property
    def current_frame_index(self) -> int:
        return self._frame_index

    @current_frame_index.setter
    def current_frame_index(self, value: int):
        n = len(self.seqs_info)
        self._frame_index = (value + n) % n
        self.message_queue = []

        info = self.seqs_info[self._frame_index]
        if info.filename is not None:
            self.enqueue_message(f'{info.filename}')
        if (note := self.read_note()) is not None:
            self.enqueue_message(note)

    @property
    def text_color(self) -> float | tuple[int, int, int]:
        return self.seqs_info[self.current_frame_index].text_color

    # ===== #
    # Notes #
    # ===== #

    def save_note(self):
        """save image-related notes to file"""
        from datetime import datetime

        fields = ['filename', 'notes', 'datetime']
        t = datetime.now().replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")  # TODO to frame dependent
        with csv_header(self.output, fields, quotes_header='notes') as csv:
            for seq in self.seqs_info:
                csv(str(seq.filename), seq.notes, t)

    def load_note(self):
        """read image-related notes from file"""
        from neuralib.util.util_verbose import printdf

        df = pl.read_csv(self.output, dtypes={'filename': pl.Utf8})
        printdf(df)
        for i, info in enumerate(self.seqs_info):
            note = df.filter(pl.col('filename') == info.filename)['notes'].item()
            self.seqs_info[i].notes = note

    def write_note(self, note: str, *, append_mode: bool = False):
        if append_mode:
            prev = self.seqs_info[self.current_frame_index].notes
            self.seqs_info[self.current_frame_index].notes = prev + ';' + note
        else:
            self.seqs_info[self.current_frame_index].notes = note

        self.current_frame_index = self.current_frame_index  # trigger enqueue_message

        if self.output is None:
            self.enqueue_message('specify output first for writing notes!')

    def read_note(self) -> str | None:
        return self.seqs_info[self._frame_index].notes

    def clear_note(self):
        self.seqs_info[self.current_frame_index].notes = None

    # ============= #
    # Key & Command #
    # ============= #

    def goto_begin(self):
        self.current_frame_index = 0

    def goto_end(self):
        self.current_frame_index = self.n_frames - 1

    def go_to(self, i: int):
        if i > len(self) or i < 0:
            self.enqueue_message(f'invalid sequence index: {i}')
            return
        self.current_frame_index = i

    def handle_keycode(self, k: int):
        # Logger.debug(f'Key: {k}')

        mapping = get_keymapping()
        ret = self._handle_keymapping(mapping, k)
        if ret is not None:  # printable
            self.buffer += chr(k)

    def _handle_keymapping(self, mapping: KeyMapping, value: int) -> int | None:
        """
        Handling the keyboard mapping
        :param mapping:
        :param value:
        :return: int value if cannot find key in keymapping, otherwise return None
        """
        ret = find_key_from_value(mapping, value)
        if not ret:
            return value

        if ret == 'left':
            self.current_frame_index -= 1
        elif ret == 'right':
            self.current_frame_index += 1
        elif ret == 'left_square_bracket':
            self.current_frame_index += 10
        elif ret == 'right_square_bracket':
            self.current_frame_index -= 10
        elif ret == 'backspace':
            if len(self.buffer) > 0:
                self.buffer = self.buffer[:-1]
        elif ret == 'enter':  # handle command in buffer
            command = self._proc_image_command = self.buffer
            self.buffer = ''
            try:
                self.handle_command(command)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                self.enqueue_message(f'command "{command}" {type(e).__name__}: {e}')
        elif ret == 'escape':
            self.buffer = ''

    def handle_command(self, command: str):
        Logger.debug(f'command: {command}')

        if command == ':h':
            self.enqueue_message(':h : print this document')
            self.enqueue_message(':q! : quit (without save)')
            self.enqueue_message(':wq : save notes and quit')
            self.enqueue_message(':c : clear current note')
            self.enqueue_message(':i : print current file index')
            self.enqueue_message(':N : goto N-th image')
            self.enqueue_message('+message : append note')
            self.enqueue_message('message : (replace) note')

        elif command == ':c':
            self.enqueue_message(f'clear notes: {self.seqs_info[self.current_frame_index].notes}')
            self.clear_note()
        elif command == ':i':
            self.enqueue_message(f'current file index: {self.current_frame_index}')
        elif re.match(r'^:(\d)', command):
            match = re.search(r'^:(\d)', command)
            self.go_to(int(match.group(1)))
        elif command.startswith('+'):
            self.write_note(command[1:], append=True)
        elif not command.startswith(':'):
            self.write_note(command)
        elif command == ':q!':
            raise KeyboardInterrupt
        elif command == ':wq':
            raise CloseSaveInterrupt
        else:
            raise RuntimeError(f'unknown command : {command}')

    # ============ #
    # Msg / Buffer #
    # ============ #

    def enqueue_message(self, text: str):
        self.message_queue.append(text)

    def _show_queued_message(self, image: np.ndarray):
        """drawing enqueued message"""
        y = 70
        s = 30
        i = 0
        while i < len(self.message_queue):
            m = self.message_queue[i]
            cv2.putText(image, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2, cv2.LINE_AA)
            i += 1
            y += s

    def _show_buffer(self, image):
        """drawing input buffer content"""
        cv2.putText(image, self.buffer, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2, cv2.LINE_AA)

    # ========= #
    # Main Loop #
    # ========= #

    def main(self):
        """main loop for the GUI"""
        cv2.namedWindow(self.window_title, cv2.WINDOW_GUI_NORMAL)

        if self.output is not None:
            if self.output.exists():
                self.load_note()

        try:
            while True:
                self._loop()
        except CloseSaveInterrupt:
            if self.output is not None:
                self.save_note()
                fprint(f'SAVE csv -> {str(self.output)}!', vtype='io')
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyWindow(self.window_title)

    def _loop(self):
        #
        try:
            info = self.seqs_info[self.current_frame_index]
        except IndexError:
            pass
        else:
            image = info.image.copy()

            if len(self.buffer):
                self._show_buffer(image)
            self._show_queued_message(image)

            cv2.imshow(self.window_title, image)

        #
        if sys.platform in ('darwin', 'linux', 'linux2'):
            k = cv2.waitKey(1)
        elif sys.platform == 'win32':
            k = cv2.waitKeyEx(1)
        else:
            raise RuntimeError('')

        if k >= 0:
            self.handle_keycode(k)


def main():
    import argparse

    ap = argparse.ArgumentParser(description='run the sequences labeller')

    ap.add_argument('-D', '--dir', type=Path, required=True, help='path with image sequences', dest='directory')
    ap.add_argument('--suffix', choices=('.pdf', '.tif', '.tiff'), default='.pdf', help='image sequence suffix')
    ap.add_argument('-O', '--output', type=Path, default=None, help='csv output for note')

    opt = ap.parse_args()

    labeller = SequenceLabeller.load_from_dir(opt.directory,
                                              file_suffix=opt.suffix,
                                              output=opt.output)

    labeller.main()


if __name__ == '__main__':
    main()
