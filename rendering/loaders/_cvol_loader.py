import struct
import numpy as np
import torch


class CVOLData(object):
    """
    Reader class for loading CVOL files to pytorch tensors

    CVOL description:
     * \brief The main storage class for volumetric datasets.
     *
     * The volume stores multiple feature channels,
     * where each feature describes, e.g., density, velocity, color.
     * Each feature is specified by the number of channel and the data type
     * (unsigned char, unsigned short, float)
     * The feature channels can have different resolutions, but are all
     * mapped to the same bounding volume.
     *
     * Format of the .cvol file (binary file):
     * <pre>
     *  [64 Bytes Header]
     *   4 bytes: magic number "CVOL"
     *   4 bytes: version (int)
     *   3*4 bytes: worldX, worldY, worldZ of type 'float'
     *     the size of the bounding box in world space
     *   4 bytes: number of features (int)
     *   4 bytes: flags (int), OR-combination of \ref Volume::Flags
     *  4 bytes: unused
     *  [Content, repeated for the number of features]
     *   4 bytes: length of the name (string)
     *  n bytes: the contents of the name string (std::string)
     *  3*8 bytes: sizeX, sizeY, sizeZ of type 'uint64_t'
     *    the voxel resolution of this feature
     *  4 bytes: number of channels (int)
     *  4 bytes: datatype (\ref DataType)
     *   Ray memory dump of the volume, sizeX*sizeY*sizeZ*channels entries of type 'datatype'.
     *     channels is fastest, followed by X and Y, Z is slowest.
     * </pre>
     *
     * Legacy format, before multi-channel support was added:
     * Format of the .cvol file (binary file):
     * <pre>
     *  [64 Bytes Header]
     *   4 bytes: magic number "cvol"
     *   3*8 bytes: sizeX, sizeY, sizeZ of type 'uint64_t'
     *   3*8 bytes: voxel size X, Y, Z in world space of type' double'
     *   4 bytes: datatype, uint value of the enum \ref Volume::DataType
     *  1 byte: bool if the volume contents are LZ4-compressed
     *   7 bytes: padding, unused
     *  [Content]
     *
     *   Ray memory dump of the volume, sizeX*sizeY*sizeZ entries of type 'datatype'.
     *   X is fastest, Z is slowest.
    """

    @staticmethod
    def _uchar_from_bytes(bytes):
        return int.from_bytes(bytes, 'little', signed=False)

    @staticmethod
    def _uint_from_bytes(bytes):
        return int.from_bytes(bytes, 'little', signed=False)

    @staticmethod
    def _short_from_bytes(bytes):
        return int.from_bytes(bytes, 'little', signed=True)

    @staticmethod
    def _float_from_bytes(bytes):
        return struct.unpack('f', bytes)[0]

    @staticmethod
    def _int_from_bytes(bytes):
        return int.from_bytes(bytes, 'little', signed=True)

    @staticmethod
    def _double_from_bytes(bytes):
        return struct.unpack('d', bytes)[0]

    def __init__(self, file_name: str, old_format=False):
        assert file_name.endswith('.cvol')
        with open(file_name, 'rb') as f:
            code = f.read(4).decode()
            self.code = code
            self.features = []
            if code == 'cvol':
                self._process_old_file_format(f)
            elif code == 'CVOL':
                self._process_new_file_format(f)
            else:
                raise Exception(f'[ERROR] Encountered unknown magic code {code}. Something seems to have gone wrong.')
            assert f.read(1) == b'', '[ERROR] Encountered trailing file contents, for which no interpretation is available'

    def _process_new_file_format(self, f):
        self._read_new_header(f)
        assert self.flags == 0, f'[ERROR] CVOL Reader does currently only support flag 0. Got {self.flags} instead.'
        for _ in range(self.num_features):
            self._read_new_feature(f)

    def _process_old_file_format(self, f):
        grid_size, data_type_code = self._read_old_header(f)
        assert self.flags == 0, f'[ERROR] CVOL Reader does currently only support flag 0. Got {self.flags} instead.'
        data = self._read_contents(f, grid_size, data_type_code)
        self.features.append({
            'name': None,
            'grid_size': grid_size,
            'num_channels': 1,
            'data': data[..., None],
        })

    def _read_old_header(self, f):
        self.version = None
        grid_size = [CVOLData._uint_from_bytes(f.read(8)) for _ in range(3)]
        voxel_size = [CVOLData._double_from_bytes(f.read(8)) for _ in range(3)]
        self.world_size = tuple(reversed([g * v for g, v in zip(grid_size, voxel_size)]))
        self.num_features = 1
        data_type_code = CVOLData._uchar_from_bytes(f.read(4))
        self.flags = int(f.read(1) != b'\x00')
        f.read(7)
        return tuple(reversed(grid_size)), data_type_code

    def _read_contents(self, f, grid_size, data_type_code):
        bytes_per_value = 2 ** data_type_code
        buffer = f.read(bytes_per_value * np.prod(grid_size))
        format = {0: 'B', 1: 'H', 2: 'f'}
        contents = memoryview(buffer).cast(format[data_type_code])
        return torch.tensor(contents).view(grid_size)

    # def _read_contents(self, f, grid_size, data_type_code):
    #     bytes_per_value = 2 ** data_type_code
    #     converter = {
    #         0: CVOLData._uchar_from_bytes,
    #         1: CVOLData._short_from_bytes,
    #         2: CVOLData._float_from_bytes,
    #     }[data_type_code]
    #     contents = [converter(f.read(bytes_per_value)) for _ in range(np.prod(grid_size))]
    #     return torch.tensor(contents).view(grid_size)

    def _read_new_header(self, f):
        self.version = CVOLData._uchar_from_bytes(f.read(4))
        self.world_size = tuple(reversed([CVOLData._float_from_bytes(f.read(4)) for _ in range(3)]))
        self.num_features = CVOLData._uchar_from_bytes(f.read(4))
        self.flags = CVOLData._uchar_from_bytes(f.read(4))
        f.read(4)

    def _read_new_feature(self, f):
        length_of_name = CVOLData._uchar_from_bytes(f.read(4))
        if length_of_name > 0:
            name = f.read(length_of_name).decode()
        else:
            name = None
        grid_size = tuple(reversed([CVOLData._uint_from_bytes(f.read(8)) for _ in range(3)]))
        num_channels = CVOLData._uchar_from_bytes(f.read(4))
        data_type_code = CVOLData._uchar_from_bytes(f.read(4))
        data = self._read_contents(f, grid_size + (num_channels,), data_type_code)
        self.features.append({
            'name': name,
            'grid_size': grid_size,
            'num_channels': num_channels,
            'data': data
        })