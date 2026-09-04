"""Accumulates and manages output image series to send back to the client."""

import copy
import logging

import ismrmrd
import numpy as np

# Return Type of process_image
ProcessImageResult = list[tuple[np.ndarray, list[ismrmrd.ImageHeader], list[ismrmrd.Meta]]]


class OutputSeries:
    """
    Manages a collection of output image series to send back to the client.

    Each series is an independent group of images with its own pixel data,
    headers, metadata, processing history, and series label. Series are
    accumulated via add() and consumed by Pipeline.run() which calls
    send_volume_as_2Dslices() for each one in insertion order.

    The series_index_offset is managed automatically, each new series
    gets a unique offset so they appear as distinct series in the client UI.

    Typical usage:

    .. code-block:: python

        series = OutputSeries(head, meta)

        # Main result
        series.add(
            data             = processed_volume,
            process_history  = ["PYTHON", "N4"],
            sequence_description = "N4",
        )

        # Optional intermediate result in a separate series
        series.add(
            data             = mask_volume,
            process_history  = ["PYTHON", "SynthstripMask"],
            sequence_description = "Brainmask",
        )

        return series.get()
    """

    def __init__(self) -> None:
        """
        Initialise the output series manager.
        """
        self.series:  list[tuple[np.ndarray, list, list]] = []
        self.n_series = 0


    def add(
        self,
        data:                 np.ndarray,
        head:                 list[ismrmrd.ImageHeader],
        meta:                 list[ismrmrd.Meta],
        process_history:      list[str] | str | None = None,
        sequence_description: list[str] | str | None = None,
    ) -> "OutputSeries":
        """
        Add an output series.

        Headers and Meta are deep-copied from the reference originals
        so each series is fully independent. The images_series_index is
        incremented automatically for each new series.

        Parameters
        ----------
        data : np.ndarray
            Processed pixel data, shape [img, cha, z, y, x], dtype int16.
        head : list of ismrmrd.ImageHeader
            Headers to associate to the series. If None, use the attribute one 
            as default.
        meta : list of ismrmrd.Meta
            Metadata to associate to the series. If None, use the attribute one 
            as default.
        
        process_history : list of str, str, or None, optional
            Processing steps to record in ImageProcessingHistory.
            Appended to any history already present in the base Meta.
            If str, wrapped in a list automatically. If None, left unchanged.
        sequence_description : list of str, str, or None, optional
            Series label shown in the client UI. If a list, joined with
            '_' (e.g. ['N4', 'Dn'] → 'N4_Dn'). If None, left unchanged.

        Returns
        -------
        OutputSeries
            Returns self to allow method chaining.
        """
        # Deep copy head and meta so series are fully independent
        if len(head) != len(meta):
            raise ValueError(
                "head and meta must have the same length, "
                "got %d and %d" % (len(head), len(meta))
            )
        if len(head) != data.shape[0]:
            raise ValueError(
                "head length (%d) must match data.shape[0] (%d)"
                % (len(head), data.shape[0])
            )

        # Deep copy so this series is fully independent
        head_copy = [copy.deepcopy(h) for h in head]
        meta_copy = [ismrmrd.Meta.deserialize(m.serialize()) for m in meta]

        # Normalise process_history
        if isinstance(process_history, str):
            process_history = [process_history]

        # Normalise sequence_description
        if isinstance(sequence_description, list):
            sequence_description = "_".join(sequence_description)

        for h, m in zip(head_copy, meta_copy):

            m["DataRole"]            = "Image"

            if process_history is not None:
                existing = m.get("ImageProcessingHistory") or []
                if isinstance(existing, str):
                    existing = [existing]
                m["ImageProcessingHistory"] = existing + process_history

            if sequence_description is not None:
                m["SequenceDescriptionAdditional"] = sequence_description

            # Incremente image_series_index
            h.image_series_index = self.n_series

        self.series.append((data, head_copy, meta_copy))
        self.n_series += 1

        logging.info(
            "OutputSeries: added series %d | description='%s' history=%s",
            self.n_series, sequence_description, process_history,
        )

        return self


    def get(self) -> ProcessImageResult:
        """
        Return all accumulated series as a list of (data, head, meta) tuples.

        Returns
        -------
        list of tuple (np.ndarray, list, list)
            All series in insertion order, ready to be consumed by
            Pipeline.run().
        """
        return self.series


    def __len__(self) -> int:
        return self.n_series


    def __repr__(self) -> str:
        return f"OutputSeries({self.n_series} series)"
