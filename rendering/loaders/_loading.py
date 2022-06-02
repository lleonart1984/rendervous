from PIL import Image
import rendering._core as ren
import struct
import numpy as np


def load_rgba_data_2D(path: str) -> ren.Buffer:
    image = Image.open(path)
    width, height = image.size
    image = image.convert("RGBA")
    data = image.getdata()
    buffer = ren.tensor(height, width, 4, dtype=ren.uint8, memory=ren.MemoryLocation.CPU, clear=False)
    arr = buffer.as_numpy().ravel()
    offset = 0
    for r, g, b, a in data:
        arr[offset + 0] = r
        arr[offset + 1] = g
        arr[offset + 2] = b
        arr[offset + 3] = a
        offset += 4
    return buffer


def _load_float_data_3D_from_xyz(path) -> ren.Buffer:
    with open(path, 'rb') as f:
        width, height, depth = struct.unpack('iii', f.read(4 * 3))
        resx, resy, resz = struct.unpack('ddd', f.read(8 * 3))
        buffer = ren.tensor(depth, height, width, dtype=ren.float32, memory=ren.MemoryLocation.CPU, clear=False)
        data = buffer.as_numpy()
        for x in range(width):
            for y in range(height):
                data[:, y, x] = struct.unpack('f' * depth, f.read(4 * depth))
    return buffer


def _load_float_data_3D_from_vol(path) -> ren.Buffer:
    with open(path, 'rb') as f:
        if f.read(3) == b'VOL':  #mitsuba vol files
            version = f.read(1)  # must be 3.
            type, = struct.unpack('i', f.read(4))  # must be 1
            width, height, depth, components = struct.unpack('iiii', f.read(4 * 4))
            assert components == 1, "Volume contains colors, not scalars"
            f.read(6*4) # BBOX
            buffer = ren.tensor(depth, height, width, dtype=ren.float32, memory=ren.MemoryLocation.CPU, clear=False)
            data = buffer.as_numpy()
            data[:,:,:] = np.ndarray(shape=[depth, height, width], dtype=np.float32, buffer=f.read(depth*height*width*4))
        else:
            mdvol_text = f.read(2) # last ol from mdvol format
            one_text = f.read(1)
            header_size, = struct.unpack('i', f.read(4))
            width, height, depth = struct.unpack('iii', f.read(4 * 3))
            resx, resy, resz = struct.unpack('fff', f.read(4 * 3))
            black, white, gamma = struct.unpack('fff', f.read(4*3))
            color_code_text = f.read(3)
            description_text = f.read(4900)
            title_text = f.read(151)
            vol_description_text = f.read(4900)
            buffer = ren.tensor(depth, height, width, dtype=ren.float32, memory=ren.MemoryLocation.CPU, clear=False)
            data = buffer.as_numpy()
            for y in range(height):
                for z in range(depth):
                    col_data = np.ndarray(width, dtype=np.uint8, buffer=f.read(width))
                    data[z, y, :] = col_data / 255.0
    return buffer


def _load_float_data_3D_from_cvol(path) -> ren.Buffer:
    from ._cvol_loader import CVOLData
    data = CVOLData(path)
    data = data.features[0]['data'][:,:,:,3].contiguous().numpy()
    depth, height, width = data.shape
    buffer = ren.tensor(depth, height, width, dtype=ren.float32, memory=ren.MemoryLocation.CPU, clear=False)
    buffer.as_numpy()[:,:,:] = data
    return buffer


def load_float_data_3D(path: str) -> ren.Buffer:
    if path.endswith('.xyz'):
        return _load_float_data_3D_from_xyz(path)
    elif path.endswith('.cvol'):
        return _load_float_data_3D_from_cvol(path)
    elif path.endswith('.vol'):
        return _load_float_data_3D_from_vol(path)
    raise Exception('Not format allowed')
