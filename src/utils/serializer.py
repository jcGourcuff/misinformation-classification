"""
IMPORTED FROM PERSONAL REPOSITORY
"""

import gzip
import json
import pickle
from typing import Any, Dict

import yaml


class ReferenceSerializer:
    """
    Embeds yaml and pkl serialization for dictionaries
    and dataframes.
    Also supports grib files and bz2 compression.
    Also, provides a way to compress (gz format).
    Everything is handled by reading the extension.
    """

    @classmethod
    def load(cls, file_path: str) -> Any:
        """
        Loads given file.

        :param file_path: Path to file.
        """
        file_name = file_path.split("/")[-1]
        formatting = cls._read_format(file_name)
        if formatting["extension"] in ["yml", "yaml"]:
            return cls._load_yaml(file_path)
        if formatting["extension"] in ["pkl", "pickle"]:
            return cls._load_pickle(
                file_path, compression=formatting["compress_format"]
            )
        if formatting["extension"] == "json":
            return cls._load_json(file_path)

        raise ValueError("Format not understood.")

    @classmethod
    def _read_format(cls, file_name: str) -> Dict[str, Any]:
        """
        Reads format of the file. Returns a dictionary with
        the following entries:
            "compress_format": Optional[str]
            "extension": str

        :param file_name: Name of file.
        """
        if file_name.startswith("."):
            extensions = file_name[1:].split(".")[1:]
        else:
            extensions = file_name.split(".")[1:]
        if len(extensions) == 1:
            return {"compress_format": None, "extension": extensions[0]}
        return {"compress_format": extensions[1], "extension": extensions[0]}

    @classmethod
    def _load_yaml(cls, file_path: str) -> Dict[Any, Any]:
        """
        Loads a yaml file.

        :param file_path: Path to file.
        """
        with open(file_path, mode="r", encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=yaml.CLoader)
            stream.close()
            return data

    @classmethod
    def _load_json(cls, file_path: str) -> Dict[str, Any]:
        """
        Loads a json file
        :param file_path:
        :return:
        """
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    @classmethod
    def _load_pickle(cls, file_path: str, compression: str | None = None) -> Any:
        """
        Loads a pickle file.

        :param file_path: Path to file.
        :param compression: If not None, must specify compression extension.
        """
        if compression is None:
            with open(file_path, "rb") as file:
                return pickle.load(file)

        assert compression == "gz"
        with gzip.open(file_path, "rb") as file:  # type: ignore
            return pickle.load(file)

    @classmethod
    def dump(cls, data: Any, file_path: str) -> None:
        """
        Dumps the data into the format specified by
        the file path.

        :param data: Data to dump.
        :param file_path: Where to dump.
        """
        file_name = file_path.split("/")[-1]
        formatting = cls._read_format(file_name)
        if formatting["extension"] in ["yml", "yaml"]:
            cls._dump_yaml(data, file_path)
        elif formatting["extension"] == "pkl":
            cls._dump_pickle(data, file_path, compression=formatting["compress_format"])
        elif formatting["extension"] == "json":
            cls._dump_json(data, file_path)
        else:
            raise ValueError("Format not understood.")

    @classmethod
    def _dump_json(cls, data: Dict[str, Any], file_path: str) -> None:
        """
        Dumps the data as json format
        :param data:
        :param file_path:
        :return:
        """
        with open(file_path, mode="w", encoding="utf-8") as stream:
            json.dump(data, stream)

    @classmethod
    def _dump_yaml(cls, data: Dict, file_path: str) -> None:
        """
        Dumps the data in yaml format.

        :param data: Data to dump.
        :param file_path: Where to dump.
        """
        with open(file_path, mode="w", encoding="utf-8") as stream:
            yaml.dump(data, stream, Dumper=yaml.CDumper)
            stream.close()

    @classmethod
    def _dump_pickle(
        cls, data: Any, file_path: str, compression: str | None = None
    ) -> None:
        """
        Dumps the data in pickle format.

        :param data: Data to dump.
        :param file_path: Where to dump.
        :param compression: Compression format.
        """
        if compression is None:
            with open(file_path, "wb") as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        else:
            assert compression == "gz"
            with gzip.open(file_path, "wb") as file:  # type: ignore
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
