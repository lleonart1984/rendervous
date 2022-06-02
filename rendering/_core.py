import itertools
import vulkan as vk
from enum import IntFlag, IntEnum
from rendering import _vkw as vkw
import math
import numpy as np
import typing
import struct
import os
import subprocess
import threading
import glm
import torch
import cupy
from cuda import cuda as cuda


def compile_shader_sources(directory='.', force_all: bool = False):
    def needs_to_update(source, binary):
        return not os.path.exists(binary) or os.path.getmtime(source) > os.path.getmtime(binary)

    for filename in os.listdir(directory):
        filename = directory + "/" + filename
        filename_without_extension, extension = os.path.splitext(filename)
        if extension == '.glsl':
            stage = os.path.splitext(filename_without_extension)[1][1:]  # [1:] for removing the dot .
            binary_file = filename_without_extension + ".spv"
            if needs_to_update(filename, binary_file) or force_all:
                if os.name == 'nt':  # Windows
                    p = subprocess.Popen(
                        os.path.expandvars(
                            '%VULKAN_SDK%/Bin/glslangValidator.exe -r -V --target-env vulkan1.2 ').replace("\\", "/")
                        + f'-S {stage} {filename} -o {binary_file}'
                    )
                else:  # Assuming Linux
                    # Quick monkeyhack for linux based distribution
                    import shlex
                    p = subprocess.Popen(
                        shlex.split(
                            os.path.expandvars('/usr/bin/glslangValidator -r -V --target-env vulkan1.2 ').replace("\\",
                                                                                                                  "/")
                            + f'-S {stage} {filename} -o {binary_file}')
                    )
                if p.wait() != 0:
                    raise RuntimeError(f"Cannot compile {filename}")
                print(f'[INFO] Compiled... {filename}')


# ------------ ALIASES ---------------


Event = vkw.Event
Window = vkw.WindowWrapper
Footprint = vkw.SubresourceFootprint
ShaderHandler = vkw.ShaderHandlerWrapper
Sampler = vkw.SamplerWrapper


# ------------- ENUMS ---------------


class PresenterMode(IntFlag):
    __no_flags_name__ = 'NONE'
    __all_flags_name__ = 'ALL'
    NONE = 0
    OFFLINE = 1
    SDL = 2


class QueueType(IntFlag):
    __no_flags_name__ = 'NONE'
    __all_flags_name__ = 'ALL'
    COPY = vk.VK_QUEUE_TRANSFER_BIT
    COMPUTE = vk.VK_QUEUE_COMPUTE_BIT
    GRAPHICS = vk.VK_QUEUE_GRAPHICS_BIT
    RAYTRACING = vk.VK_QUEUE_GRAPHICS_BIT


class BufferUsage(IntFlag):
    NONE = 0
    VERTEX = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    INDEX = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    UNIFORM = vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    STORAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    TRANSFER_SRC = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    TRANSFER_DST = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    GPU_ADDRESS = vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    RAYTRACING_ADS = vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
    RAYTRACING_ADS_READ = vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    TENSOR = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT


class ImageUsage(IntFlag):
    NONE = 0
    TRANSFER_SRC = vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    TRANSFER_DST = vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
    RENDER_TARGET = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    SAMPLED = vk.VK_IMAGE_USAGE_SAMPLED_BIT
    DEPTH_STENCIL = vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    STORAGE = vk.VK_IMAGE_USAGE_STORAGE_BIT


class Format(IntEnum):
    NONE = vk.VK_FORMAT_UNDEFINED
    UINT_RGBA = vk.VK_FORMAT_R8G8B8A8_UINT
    UINT_RGB = vk.VK_FORMAT_R8G8B8_UINT
    UINT_BGRA_STD = vk.VK_FORMAT_B8G8R8A8_SRGB
    UINT_RGBA_STD = vk.VK_FORMAT_R8G8B8A8_SRGB
    UINT_RGBA_UNORM = vk.VK_FORMAT_R8G8B8A8_UNORM
    UINT_BGRA_UNORM = vk.VK_FORMAT_B8G8R8A8_UNORM
    FLOAT = vk.VK_FORMAT_R32_SFLOAT
    INT = vk.VK_FORMAT_R32_SINT
    UINT = vk.VK_FORMAT_R32_UINT
    VEC2 = vk.VK_FORMAT_R32G32_SFLOAT
    VEC3 = vk.VK_FORMAT_R32G32B32_SFLOAT
    VEC4 = vk.VK_FORMAT_R32G32B32A32_SFLOAT
    IVEC2 = vk.VK_FORMAT_R32G32_SINT
    IVEC3 = vk.VK_FORMAT_R32G32B32_SINT
    IVEC4 = vk.VK_FORMAT_R32G32B32A32_SINT
    UVEC2 = vk.VK_FORMAT_R32G32_UINT
    UVEC3 = vk.VK_FORMAT_R32G32B32_UINT
    UVEC4 = vk.VK_FORMAT_R32G32B32A32_UINT
    PRESENTER = vk.VK_FORMAT_B8G8R8A8_SRGB


class ImageType(IntEnum):
    TEXTURE_1D = vk.VK_IMAGE_TYPE_1D
    TEXTURE_2D = vk.VK_IMAGE_TYPE_2D
    TEXTURE_3D = vk.VK_IMAGE_TYPE_3D


class MemoryLocation(IntFlag):
    """
    Memory configurations
    """
    """
    Efficient memory for reading and writing on the GPU.
    """
    GPU = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    """
    Memory can be read and write directly from the CPU
    """
    CPU = vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT


class ShaderStage(IntEnum):
    VERTEX = vk.VK_SHADER_STAGE_VERTEX_BIT
    FRAGMENT = vk.VK_SHADER_STAGE_FRAGMENT_BIT
    COMPUTE = vk.VK_SHADER_STAGE_COMPUTE_BIT
    RT_GENERATION = vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR
    RT_CLOSEST_HIT = vk.VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
    RT_MISS = vk.VK_SHADER_STAGE_MISS_BIT_KHR
    RT_ANY_HIT = vk.VK_SHADER_STAGE_ANY_HIT_BIT_KHR
    RT_INTERSECTION_HIT = vk.VK_SHADER_STAGE_INTERSECTION_BIT_KHR


class Filter(IntEnum):
    POINT = vk.VK_FILTER_NEAREST
    LINEAR = vk.VK_FILTER_LINEAR


class MipMapMode(IntEnum):
    POINT = vk.VK_SAMPLER_MIPMAP_MODE_NEAREST
    LINEAR = vk.VK_SAMPLER_MIPMAP_MODE_LINEAR


class AddressMode(IntEnum):
    REPEAT = vk.VK_SAMPLER_ADDRESS_MODE_REPEAT
    CLAMP_EDGE = vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
    BORDER_COLOR = vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER


class CompareOp(IntEnum):
    NEVER = vk.VK_COMPARE_OP_NEVER
    LESS = vk.VK_COMPARE_OP_LESS,
    EQUAL = vk.VK_COMPARE_OP_EQUAL,
    LESS_OR_EQUAL = vk.VK_COMPARE_OP_LESS_OR_EQUAL,
    GREATER = vk.VK_COMPARE_OP_GREATER,
    NOT_EQUAL = vk.VK_COMPARE_OP_NOT_EQUAL,
    GREATER_OR_EQUAL = vk.VK_COMPARE_OP_GREATER_OR_EQUAL,
    ALWAYS = vk.VK_COMPARE_OP_ALWAYS,


class BorderColor(IntEnum):
    TRANSPARENT_BLACK_FLOAT = vk.VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    TRANSPARENT_BLACK_INT = vk.VK_BORDER_COLOR_INT_TRANSPARENT_BLACK,
    OPAQUE_BLACK_FLOAT = vk.VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    OPAQUE_BLACK_INT = vk.VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    OPAQUE_WHITE_FLOAT = vk.VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    OPAQUE_WHITE_INT = vk.VK_BORDER_COLOR_INT_OPAQUE_WHITE


dtype = torch.dtype

float32: dtype = torch.float32
_float: dtype = torch.float
float64: dtype = torch.float64
double: dtype = torch.double
float16: dtype = torch.float16
bfloat16: dtype = torch.bfloat16
half: dtype = torch.half
uint8: dtype = torch.uint8
int8: dtype = torch.int8
int16: dtype = torch.int16
short: dtype = torch.short
int32: dtype = torch.int32
_int: dtype = torch.int
int64: dtype = torch.int64
long: dtype = torch.long
complex32: dtype = torch.complex32
complex64: dtype = torch.complex64
cfloat: dtype = torch.cfloat
complex128: dtype = torch.complex128
cdouble: dtype = torch.cdouble
quint8: dtype = torch.quint8
qint8: dtype = torch.qint8
qint32: dtype = torch.qint32
_bool: dtype = torch.bool
quint4x2: dtype = torch.quint4x2


# ---- HIGH LEVEL DEFINITIONS ----------


class Interop:
    __TYPE_SIZES = {
        int: 4,
        float: 4,  # not really but more legible in graphics context

        glm.uint64: 8,

        glm.vec1: 4,
        glm.float32: 4,
        glm.vec2: 4 * 2,
        glm.vec3: 4 * 3,
        glm.vec4: 4 * 4,

        glm.ivec1: 4 * 1,
        glm.int32: 4 * 1,
        glm.ivec2: 4 * 2,
        glm.ivec3: 4 * 3,
        glm.ivec4: 4 * 4,

        glm.uint32: 4 * 1,
        glm.uvec1: 4 * 1,
        glm.uvec2: 4 * 2,
        glm.uvec3: 4 * 3,
        glm.uvec4: 4 * 4,

        glm.mat2x2: 4 * 2 * 2,
        glm.mat2x3: 4 * 2 * 3,
        glm.mat2x4: 4 * 2 * 4,

        glm.mat3x2: 4 * 3 * 2,
        glm.mat3x3: 4 * 3 * 3,
        glm.mat3x4: 4 * 3 * 4,

        glm.mat4x2: 4 * 4 * 2,
        glm.mat4x3: 4 * 4 * 3,
        glm.mat4x4: 4 * 4 * 4,
    }

    @staticmethod
    def size_of(t):
        assert t in Interop.__TYPE_SIZES, f"Not supported type {t}, use int, float or glm types"
        return Interop.__TYPE_SIZES[t]

    @staticmethod
    def to_bytes(t, value):
        assert t in Interop.__TYPE_SIZES, f"Not supported type {t}, use int, float or glm types"
        if t == int:
            return struct.pack('i', value)
        if t == float:
            return struct.pack('f', value)
        if t == glm.uint64:
            return struct.pack('<Q', value.value)
        return t.to_bytes(value)

    @staticmethod
    def from_bytes(t, buffer):
        assert t in Interop.__TYPE_SIZES, f"Not supported type {t}, use int, float or glm types"
        if t == int:
            return struct.unpack('i', buffer)[0]
        if t == float:
            return struct.unpack('f', buffer)[0]
        if t == glm.uint64:
            return struct.unpack('Q', buffer)[0]
        if isinstance(buffer, bytearray):
            return t.from_bytes(bytes(buffer))
        return t.from_bytes(vk.ffi.buffer(buffer)[0:Interop.size_of(t)])

    __TORCH_DTYPE_TO_CUPY__ = {
        torch.float32: cupy.float32,
        torch.float: cupy.float32,
        torch.float64: cupy.float64,
        torch.double: cupy.float64,
        torch.float16: cupy.float16,
        torch.bfloat16: cupy.float16,
        torch.half: cupy.float16,
        torch.uint8: cupy.uint8,
        torch.int8: cupy.int8,
        torch.int16: cupy.int16,
        torch.short: cupy.int16,
        torch.int32: cupy.int32,
        torch.int: cupy.int32,
        torch.int64: cupy.int64,
        torch.long: cupy.int64,
        torch.complex32: cupy.float16,
        torch.complex64: cupy.complex64,
        torch.cfloat: cupy.complex64,
        torch.complex128: cupy.complex128,
        torch.cdouble: cupy.complex128,
        torch.quint8: cupy.ubyte,
        torch.qint8: cupy.byte,
        torch.qint32: cupy.int32,
        torch.bool: cupy.bool8,
        torch.quint4x2: cupy.uint8
    }

    __TORCH_DTYPE_TO_NUMPY__ = {
        torch.float32: np.float32,
        torch.float: np.float32,
        torch.float64: np.double,
        torch.double: np.double,
        torch.float16: np.half,
        torch.bfloat16: np.half,
        torch.half: np.half,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.short: np.int16,
        torch.int32: np.int32,
        torch.int: np.int32,
        torch.int64: np.int64,
        torch.long: np.int64,
        torch.complex32: np.half,
        torch.complex64: np.singlecomplex,
        torch.cfloat: np.csingle,
        torch.complex128: np.longcomplex,
        torch.cdouble: np.cdouble,
        torch.quint8: np.ubyte,
        torch.qint8: np.byte,
        torch.qint32: np.int32,
        torch.bool: np.char,
        torch.quint4x2: np.char
    }

    __CUPY_DTYPE_TO_TORCH__ = {
        cupy.float32: torch.float32,
        cupy.float64: torch.float64,
        cupy.int: torch.int32,
        cupy.int32: torch.int32,
        cupy.uint32: torch.int32,
        cupy.uint8: torch.uint8,
        cupy.byte: torch.int8,
        cupy.uint64: torch.long,
        cupy.int64: torch.int64,
    }

    __NUMPY_DTYPE_TO_TORCH__ = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.uint32: torch.int32,
        np.uint8: torch.uint8,
        np.byte: torch.int8,
        np.uint64: torch.long,
        np.int64: torch.int64,
    }

    @staticmethod
    def dtype_to_numpy(torch_dtype):
        return Interop.__TORCH_DTYPE_TO_NUMPY__[torch_dtype]

    @staticmethod
    def dtype_to_cupy(torch_dtype):
        return Interop.__TORCH_DTYPE_TO_CUPY__[torch_dtype]

    @staticmethod
    def dtype_from_numpy(numpy_dtype):
        return Interop.__NUMPY_DTYPE_TO_TORCH__[numpy_dtype]

    @staticmethod
    def dtype_from_cupy(cupy_dtype):
        return Interop.__CUPY_DTYPE_TO_TORCH__[cupy_dtype]

    __FORMAT_TO_TORCH_INFO__ = {
        Format.NONE: (1, torch.uint8),
        Format.UINT_RGBA: (4, torch.uint8),
        Format.UINT_RGB: (3, torch.uint8),
        Format.UINT_BGRA_STD: (4, torch.uint8),
        Format.UINT_RGBA_STD: (4, torch.uint8),
        Format.UINT_RGBA_UNORM: (4, torch.uint8),
        Format.UINT_BGRA_UNORM: (4, torch.uint8),
        Format.FLOAT: (1, torch.float32),
        Format.INT: (1, torch.int32),
        Format.UINT: (1, torch.int32),
        Format.VEC2: (2, torch.float32),
        Format.VEC3: (3, torch.float32),
        Format.VEC4: (4, torch.float32),
        Format.IVEC2: (2, torch.int32),
        Format.IVEC3: (3, torch.int32),
        Format.IVEC4: (4, torch.int32),
        Format.UVEC2: (2, torch.int32),
        Format.UVEC3: (3, torch.int32),
        Format.UVEC4: (2, torch.int32)
    }

    @staticmethod
    def info_from_format(format: Format):
        return Interop.__FORMAT_TO_TORCH_INFO__[format]

    @staticmethod
    def size_of_dtype(dtype: dtype):
        return torch.tensor([], dtype=dtype).element_size()


class Resource(object):
    def __init__(self, device, w_resource: vkw.ResourceWrapper):
        self.w_resource = w_resource
        self.device = device
        self._info = dict()

    def is_buffer(self):
        return self.w_resource.resource_data.is_buffer

    def is_host_visible(self):
        return self.w_resource.resource_data.is_cpu_visible

    def is_on_gpu(self):
        return self.w_resource.resource_data.is_gpu

    def get_footprints(self):
        return self.w_resource.get_footprints()

    def is_single(self):
        return self.w_resource.is_single()

    # def is_homogeneous(self):
    #     return self.w_resource.is_homogeneous()

    def load(self, src_data):
        if src_data is self:
            return
        if isinstance(src_data, Resource):
            src_data = src_data.w_resource
        self.w_resource.load(self.device.w_device, src_data)
        return self

    def save(self, dst_data):
        if dst_data is self:
            return
        if isinstance(dst_data, Resource):
            dst_data = dst_data.w_resource
        self.w_resource.save(self.device.w_device, dst_data)
        return self

    def get_shape(self):
        if 'shape' in self._info:
            return self._info['shape']
        return [(self.w_resource.get_size(),)]

    def get_dtype(self):
        if 'dtype' in self._info:
            return self._info['dtype']
        return torch.uint8

    def as_raw_tensor(self) -> torch.Tensor:
        return self.w_resource.get_tensor()

    def storage_tensor(self) -> torch.Tensor:
        return self.w_resource.resource_data.w_memory.get_tensor()

    def as_tensor(self) -> torch.Tensor:
        assert self.is_buffer, "Only homogeneous resource slice can be seen as single tensor"
        if self.is_on_gpu():
            datas = self.as_cupy()
            device = torch.device('cuda:0')
        else:
            datas = self.as_numpy()
            device = torch.device('cpu')
        if not isinstance(datas, list):
            datas = [datas]
        tensors = [torch.as_tensor(d, device=device) for d in datas]
        # Mark as vulkanized
        for t in tensors:
            t.is_vulkanized = True
            t._back_buffer = self
        return tensors[0] if len(tensors) == 1 else tensors

    def _split(self, a):
        subresources = []
        offset = 0
        for f in self.get_shape():
            shape = f
            elements = math.prod(shape)
            slice = a[offset:offset + elements]
            subresources.append(slice.reshape(*shape))
            offset += elements
        return subresources[0] if len(subresources) == 1 else subresources

    def as_cupy(self):
        assert self.is_buffer() and self.is_on_gpu(), "Only homogeneous resource slice on gpu can be seen as single cupy array"
        dtype = self.get_dtype()
        cupy_dtype: cupy.dtype = Interop.dtype_to_cupy(dtype)()
        size = self.w_resource.get_size() // cupy_dtype.itemsize
        full_array_shape = (size,)
        a = cupy.ndarray(
            shape=full_array_shape,
            dtype=cupy_dtype,
            memptr=cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(self.w_resource.get_cuda_ptr(), self.w_resource.get_size(), self, 0), 0)
        )
        return self._split(a)

    def as_numpy(self, move_to_cpu=False):
        assert self.is_buffer() and (self.is_host_visible() or move_to_cpu), "Only homogeneous resource slice on cpu can be seen as single numpy array"
        dtype = self.get_dtype()
        np_dtype = Interop.dtype_to_numpy(dtype)
        a = self.as_raw_tensor().cpu().numpy().view(np_dtype)
        return self._split(a)

    def __str__(self):
        if self.is_buffer():
            return str(self.as_tensor())
        return "Image (missing description)"

        # gpu_tensor = self.as_gpu_tensor()
        # return str(gpu_tensor)+'\n'+str(gpu_tensor.min())+'\n'+str(gpu_tensor.max())

    def __repr__(self):
        return str(self)


class Buffer(Resource):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper):
        super().__init__(device, w_buffer)
        self.size = w_buffer.current_slice["size"]

    def data_ptr(self):
        return self.w_resource.get_device_ptr()

    def cuda_ptr(self):
        return self.w_resource.resource_data.w_memory.cuda_ptr() + self.w_resource.current_slice['offset']

    def get_element_size(self):
        if 'element_size' in self._info:
            return self._info['element_size']
        return 1

    def get_numel(self):
        return self.size // self.get_element_size()

    def raw(self):
        if 'dtype' not in self._info:
            return self
        return self.slice(0, self.size)

    def reinterpret(self, dtype: dtype = float32):
        assert self.get_element_size() == 1, "Only raw buffers can be reinterpret"
        b = self.slice(0, self.size)
        element_size = Interop.size_of_dtype(dtype)
        assert self.size % element_size == 0, "Element size must divide buffer size"
        b._info['shape'] = [(self.size // element_size,)]
        b._info['element_size'] = element_size
        b._info['dtype'] = dtype
        return b

    def as_floats(self):
        return self.raw().reinterpret()

    def as_ints(self):
        return self.raw().reinterpret(torch.int)

    def as_bytes(self):
        return self.raw()

    def reshape(self, *sizes):
        def fix_shape(shape):
            shape = list(shape)
            num_el = 1
            free_pos = -1
            for i, s in enumerate(shape):
                assert s == -1 or s > 0, "Dimensions must be positive or -1 to indicate one of the dimensions as variable"
                if s == -1:
                    assert free_pos == -1, "Not possible two variable dimensions"
                    free_pos = i
                else:
                    num_el *= s
            assert (
                               self.size // self.get_element_size()) % num_el == 0, "Dimensions are not valid combination for the number of elements of this buffer"
            if free_pos != -1:
                shape[free_pos] = (self.size // self.get_element_size()) // num_el
            return shape

        sizes = fix_shape(sizes)
        b = self.slice(0, self.size)
        b._info = dict(self._info)
        b._info['shape'] = [sizes]
        return b

    def reshape_as(self, image: Resource):
        assert not image.is_buffer(), "Can only reshape a buffer internally to image subresources"
        image_cpu_size = image.w_resource.resource_data.cpu_size
        assert self.size >= image_cpu_size, "Can not reshape this buffer to the image because the size is not sufficient"
        b = self.slice(0, image_cpu_size)
        shapes = []
        components, dtype = Interop.info_from_format(image.w_resource.resource_data.vk_description.format)
        for f in image.get_footprints():
            width, height, depth = f.dim
            d = image.get_image_dimension()
            if d == 1:
                shape = (width, components)
            elif d == 2:
                shape = (height, width, components)
            else:
                shape = (depth, height, width, components)
            shapes.append(shape)
        b._info['shape'] = shapes
        b._info['dtype'] = dtype
        return b

    # def as_cupy(self):
    #     dtype = Interop.dtype_to_cupy(self.get_type())()
    #     shape = self.get_shape()
    #     return cupy.ndarray(
    #         shape=shape,
    #         dtype=dtype,
    #         memptr=cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(self.cuda_ptr(), self.size, self, 0), 0)
    #     )

    def slice(self, offset, size):
        return Buffer(self.device, self.w_resource.slice_buffer(offset, size))

    def structured(self, **fields):
        layout, size = Uniform.process_layout(fields)
        return StructuredBuffer(self.device, self.w_resource, layout, size)

    def as_indices(self):
        return IndexBuffer(self.device, self.w_resource)

    def clone(self, memory: MemoryLocation) -> 'Buffer':
        return self.device.create_tensor_like(self, memory, clear=False).load(self)


__SKIP_ATTR__ = {"layout", "w_resource", "size", "device", "_info"}


class Uniform(Buffer):

    @staticmethod
    def process_layout(fields: typing.Dict[str, type]):
        offset = 0
        layout = {}
        for field, t in fields.items():
            field_size = Interop.size_of(t)
            layout[field] = (offset, field_size, t)
            offset += field_size
        return layout, offset

    def __init__(self, device, w_buffer: vkw.ResourceWrapper, layout: typing.Dict[str, typing.Tuple[int, int, type]]):
        self.layout = layout
        super().__init__(device, w_buffer)

    def __getattr__(self, item):
        if item in __SKIP_ATTR__:
            return super(Uniform, self).__getattribute__(item)
        if item not in self.layout:
            return super(Uniform, self).__getattribute__(item)
        offset, size, t = self.layout[item]
        buffer = bytearray(size)
        self.slice(offset, size).save(buffer)
        return Interop.from_bytes(t, buffer)

    def __setattr__(self, item, value):
        if item in __SKIP_ATTR__:
            super(Uniform, self).__setattr__(item, value)
            return
        if item in self.layout:
            offset, size, t = self.layout[item]
            buffer = Interop.to_bytes(t, value)
            self.slice(offset, size).load(buffer)
            return
        raise AttributeError(f"Attribute {item} doesnt exist in this uniform")


class WrappedPtr(Uniform):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper):
        super().__init__(device, w_buffer, {'ptr': (0, 8, glm.uint64)})
        self.ptr = glm.uint64(0)
        self._info['dtype'] = torch.int64
        self._info['shape'] = [(1,)]
        self._info['element_size'] = 8

    def wrap(self, obj, offset: int = 0):
        owner = None
        ptr = 0
        if obj is None:
            offset = 0
        if isinstance(obj, torch.Tensor):
            assert obj.is_cuda, "Only gpu tensors can be wrapped"
            owner = self.device.as_buffer(obj)
            ptr = owner.data_ptr()
        elif isinstance(obj, Buffer):
            owner = obj
            ptr = owner.data_ptr()
        self.ptr = glm.uint64(obj + offset)
        self._info['ptr_owner'] = owner

    def unwrap(self):
        self.ptr = glm.uint64(0)
        self._info['ptr_owner'] = None


class WrappedPtrCollection(Buffer):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper, ptr_count: int):
        super().__init__(device, w_buffer)
        self._info['count'] = ptr_count
        self._info['owners'] = [None] * ptr_count
        self._info['dtype'] = torch.int64
        self._info['shape'] = [(ptr_count,)]
        self._info['element_size'] = 8

    def wrap(self, *objs, start_index=0):
        for i, obj in enumerate(objs):
            if i + start_index >= self._info['count']:
                raise Exception('Object fall outside valid number of ptrs for this collection')
            owner = None
            ptr = 0
            if obj is not None:
                if isinstance(obj, torch.Tensor):
                    assert obj.is_cuda, "Only gpu tensors can be wrapped"
                    owner = self.device.as_buffer(obj)
                    ptr = owner.data_ptr()
                elif isinstance(obj, Buffer):
                    owner = obj
                    ptr = owner.data_ptr()
            bytes = Interop.to_bytes(glm.uint64, glm.uint64(ptr))
            self.w_resource.slice_buffer((i + start_index) * 8, 8).load(self.device.w_device, bytes)
            self._info['owners'][i + start_index] = owner

    def wrap_model_params(self, model):
        self.wrap(*model.parameters())

    def wrap_model_grads(self, model):
        for p in model.parameters():
            if p.grad is None:
                p.grad = self.device.create_tensor_like(p).as_tensor()
        self.wrap(*tuple(p.grad for p in model.parameters()))

    def unwrap(self):
        null_bytes = Interop.to_bytes(glm.uint64, glm.uint64(0))
        for i in range(self._info['count']):
            self._info['owners'][i] = None
            self.w_resource.slice_buffer(i * 8, 8).load(self.device.w_device, null_bytes)


class StructuredBuffer(Buffer):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper, layout: typing.Dict[str, typing.Tuple[int, int, type]], stride: int):
        self.layout = layout
        super().__init__(device, w_buffer)
        self.stride = stride
        self._info['element_size'] = stride

    def __getitem__(self, item):
        return Uniform(self.device, self.w_resource.slice_buffer(item * self.stride, self.stride), self.layout)


class IndexBuffer(Buffer):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper):
        super().__init__(device, w_buffer)
        self._info['dtype'] = torch.int32
        self._info['element_size'] = 4
        self._info['shape'] = [(self.size // 4,)]

    def __getitem__(self, item):
        bytes = bytearray(4)
        self.w_resource.slice_buffer(item * 4, 4).save(bytes)
        return struct.unpack('i', bytes)[0]

    def __setitem__(self, key, value):
        assert isinstance(value, int) or isinstance(value, glm.uint32), "Only integers is supported"
        bytes = struct.pack('i', value)
        self.w_resource.slice_buffer(key * 4, 4).load(self.device.w_device, bytes)


class Image(Resource):
    @staticmethod
    def compute_dimension(width: int, height: int, depth: int, mip_level: int):
        return max(1, width // (1 << mip_level)), max(1, height // (1 << mip_level)), max(1, depth // (1 << mip_level))

    def __init__(self, device, w_image: vkw.ResourceWrapper):
        super().__init__(device, w_image)
        self.width, self.height, self.depth = Image.compute_dimension(
            w_image.resource_data.vk_description.extent.width,
            w_image.resource_data.vk_description.extent.height,
            w_image.resource_data.vk_description.extent.depth,
            w_image.current_slice["mip_start"]
        )

    def get_image_dimension(self) -> int:
        type = self.w_resource.resource_data.vk_description.imageType
        return type + 1

    def get_mip_count(self) -> int:
        return self.w_resource.current_slice["mip_count"]

    def get_layer_count(self) -> int:
        return self.w_resource.current_slice["array_count"]

    def slice_mips(self, mip_start, mip_count):
        return Image(self.device, self.w_resource.slice_mips(mip_start, mip_count))

    def slice_array(self, array_start, array_count):
        return Image(self.device, self.w_resource.slice_array(array_start, array_count))

    def subresource(self, mip=0, layer=0):
        return Image(self.device, self.w_resource.subresource(mip, layer))

    def as_readonly(self):
        return Image(self.device, self.w_resource.as_readonly())


class GeometryCollection:

    def __init__(self, device: vkw.DeviceWrapper):
        self.w_device = device
        self.descriptions = []

    def __del__(self):
        self.w_device = None
        self.descriptions = []

    def get_collection_type(self) -> int:
        pass


class TriangleCollection(GeometryCollection):

    def __init__(self, device: vkw.DeviceWrapper):
        super().__init__(device)

    def append(self, vertices: StructuredBuffer,
               indices: IndexBuffer = None,
               transform: StructuredBuffer = None):
        assert transform is None or transform.stride == 4 * 12, "Transform buffer can not be cast to 3x4 float"
        self.descriptions.append((vertices, indices, transform))

    def get_collection_type(self) -> int:
        return 0  # Triangles


class Instance:
    def __init__(self, instance_buffer, index, tensor_buffer):
        self.instance_buffer = instance_buffer
        self.index = index
        self.tensor_buffer = tensor_buffer

    def __get_int_value(self, offset, size):
        bytes = bytearray(4)
        bytes[0:size] = self.tensor_buffer[offset:offset + size]
        return struct.unpack('i', bytes)[0]

    def __set_int_value(self, offset, size, value):
        bytes = struct.pack('i', value)
        self.tensor_buffer[offset:offset + size] = torch.frombuffer(bytes[0:size], dtype=torch.uint8)

    def _get_transform(self):
        return Interop.from_bytes(glm.mat3x4, self.tensor_buffer[0:48])

    def _set_transform(self, value):
        self.tensor_buffer[0:48] = torch.frombuffer(Interop.to_bytes(glm.mat3x4, value), dtype=torch.uint8)

    transform = property(fget=_get_transform, fset=_set_transform)

    # def _get_mask(self):
    #     return self.__get_int_value(48, 1)
    #
    # def _set_mask(self, value):
    #     self.__set_int_value(48, 1, value)
    #
    # mask = property(fget=_get_mask, fset=_set_mask)
    #
    #
    # def _get_id(self):
    #     return self.__get_int_value(49, 3)
    #
    # def _set_id(self, value):
    #     self.__set_int_value(49, 3, value)
    #
    # id = property(fget=_get_id, fset=_set_id)
    #
    #
    # def _get_flags(self):
    #     return self.__get_int_value(52, 1)
    #
    # def _set_flags(self, value):
    #     self.__set_int_value(52, 1, value)
    #
    # flags = property(fget=_get_flags, fset=_set_flags)
    #
    # def _get_offset(self):
    #     return self.__get_int_value(53, 3)
    #
    # def _set_offset(self, value):
    #     self.__set_int_value(53, 3, value)
    #
    # offset = property(fget=_get_offset, fset=_set_offset)

    def _get_id(self):
        return self.__get_int_value(48, 3)

    def _set_id(self, value):
        self.__set_int_value(48, 3, value)

    id = property(fget=_get_id, fset=_set_id)

    def _get_mask(self):
        return self.__get_int_value(51, 1)

    def _set_mask(self, value):
        self.__set_int_value(51, 1, value)

    mask = property(fget=_get_mask, fset=_set_mask)

    def _get_offset(self):
        return self.__get_int_value(52, 3)

    def _set_offset(self, value):
        self.__set_int_value(52, 3, value)

    offset = property(fget=_get_offset, fset=_set_offset)

    def _get_flags(self):
        return self.__get_int_value(55, 1)

    def _set_flags(self, value):
        self.__set_int_value(55, 1, value)

    flags = property(fget=_get_flags, fset=_set_flags)

    def _get_geometry(self):
        return self.instance_buffer.geometries[self.index]

    def _set_geometry(self, geometry):
        self.instance_buffer.geometries[self.index] = geometry
        self.tensor_buffer[56:64] = torch.frombuffer(struct.pack('Q', geometry.handle), dtype=torch.uint8)

    geometry = property(fget=_get_geometry, fset=_set_geometry)


class InstanceBuffer(Resource):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper, instances: int):
        super().__init__(device, w_buffer)
        self.buffer : vkw.ResourceWrapper = self.w_resource
        self.number_of_instances = instances
        self.geometries = [None] * instances  # list to keep referenced geometry's wrappers alive

    def __getitem__(self, item):
        return Instance(self, item, self.buffer.get_tensor()[item * 64:item * 64 + 64])


class ADS(Resource):
    def __init__(self, device, w_resource: vkw.ResourceWrapper, handle, scratch_size,
                 info: vk.VkAccelerationStructureCreateInfoKHR, ranges, instance_buffer=None):
        super().__init__(device, w_resource)
        self.ads = w_resource.resource_data.ads
        self.ads_info = info
        self.handle = handle
        self.scratch_size = scratch_size
        self.ranges = ranges
        self.instance_buffer = instance_buffer


class RTProgram:

    def __init__(self, pipeline,
                 w_shader_table: vkw.ResourceWrapper,
                 miss_offset,
                 hit_offset):
        self.pipeline = pipeline
        prop = self.pipeline.w_pipeline.w_device.raytracing_properties
        self.shader_handle_stride = prop.shaderGroupHandleSize
        self.table_buffer = w_shader_table.get_tensor()
        self.w_table = w_shader_table
        self.raygen_slice = w_shader_table.slice_buffer(0, self.shader_handle_stride)
        self.miss_slice = w_shader_table.slice_buffer(miss_offset, hit_offset - miss_offset)
        self.hitgroup_slice = w_shader_table.slice_buffer(hit_offset, w_shader_table.get_size() - hit_offset)
        self.miss_offset = miss_offset
        self.hit_offset = hit_offset

    def __del__(self):
        self.pipeline = None
        self.w_table = None
        self.raygen_slice = None
        self.miss_slice = None
        self.hitgroup_slice = None

    def set_generation(self, shader_group: ShaderHandler):
        self.table_buffer[0:self.shader_handle_stride] = torch.frombuffer(shader_group.get_handle(), dtype=torch.uint8)

    def set_miss(self, miss_index: int, shader_group: ShaderHandler):
        self.table_buffer[
        self.miss_offset + self.shader_handle_stride * miss_index:
        self.miss_offset + self.shader_handle_stride * (miss_index + 1)] = \
            torch.frombuffer(shader_group.get_handle(), dtype=torch.uint8)

    def set_hit_group(self, hit_group_index, shader_group: ShaderHandler):
        self.table_buffer[
        self.hit_offset + self.shader_handle_stride * hit_group_index:
        self.hit_offset + self.shader_handle_stride * (hit_group_index + 1)] = \
            torch.frombuffer(shader_group.get_handle(), dtype=torch.uint8)


class Pipeline:
    def __init__(self, w_pipeline: vkw.PipelineBindingWrapper):
        self.w_pipeline = w_pipeline

    def __setup__(self):
        pass

    def is_closed(self):
        return self.w_pipeline.initialized

    def close(self):
        self.w_pipeline._build_objects()

    def descriptor_set(self, set_slot: int):
        self.w_pipeline.descriptor_set(set_slot)

    class _ShaderStageContext:

        def __init__(self, pipeline, new_stage):
            self.pipeline = pipeline
            self.new_stage = new_stage
            self.old_stage = pipeline.w_pipeline.active_stage

        def __enter__(self):
            self.pipeline.w_pipeline.shader_stage(self.new_stage)
            return self.pipeline

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pipeline.w_pipeline.shader_stage(self.old_stage)

    def shader_stage(self, shader_stage: ShaderStage):
        return Pipeline._ShaderStageContext(self, shader_stage)

    def _bind_resource(self, slot: int, count: int, resolver, descriptor_type):
        self.w_pipeline.binding(
            slot=slot,
            vk_descriptor_type=descriptor_type,
            count=count,
            resolver=resolver
        )

    def bind_uniform(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)

    def bind_uniform_array(self, slot: int, count: int, resolver):
        self._bind_resource(slot, count, resolver, vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)

    def bind_wrap_gpu(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)

    def bind_wrap_gpu_collection(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)

    def bind_wrap_gpu_array(self, slot: int, count: int, resolver):
        self._bind_resource(slot, count, resolver, vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)

    def bind_storage_buffer(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)

    def bind_storage_buffer_array(self, slot: int, count: int, resolver):
        self._bind_resource(slot, count, resolver, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)

    def bind_texture_combined(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)

    def bind_texture_combined_array(self, slot: int, count: int, resolver):
        self._bind_resource(slot, count, resolver, vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)

    def bind_texture(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)

    def bind_texture_array(self, slot: int, count: int, resolver):
        self._bind_resource(slot, count, resolver, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)

    def bind_storage_image(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)

    def bind_storage_image_array(self, slot: int, count: int, resolver):
        self._bind_resource(slot, count, resolver, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)

    def bind_scene_ads(self, slot: int, resolver):
        self._bind_resource(slot, 1, lambda: [resolver()], vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)

    def bind_constants(self, offset: int, **fields):
        layout, size = Uniform.process_layout(fields)
        self.w_pipeline.add_constant_range(offset, size, layout)

    def load_shader(self, path, *specialization, main_function = 'main'):
        return self.w_pipeline.load_shader(vkw.ShaderStageWrapper.from_file(
            device=self.w_pipeline.w_device,
            main_function=main_function,
            path=path,
            specialization=specialization
        ))

    # def load_fragment_shader(self, path: str, *specialization, main_function='main'):
    #     return self.load_shader(vk.VK_SHADER_STAGE_FRAGMENT_BIT, path, main_function, *specialization)
    #
    # def load_vertex_shader(self, path: str, *specialization, main_function='main'):
    #     return self.load_shader(vk.VK_SHADER_STAGE_VERTEX_BIT, path, main_function, *specialization)
    #
    # def load_compute_shader(self, path: str, *specialization, main_function='main'):
    #     return self.load_shader(vk.VK_SHADER_STAGE_COMPUTE_BIT, path, main_function, specialization)
    #
    # def load_rt_generation_shader(self, path: str, *specialization, main_function='main'):
    #     return self.load_shader(vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR, path, main_function, specialization)
    #
    # def load_rt_closest_hit_shader(self, path: str, *specialization, main_function='main'):
    #     return self.load_shader(vk.VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, path, main_function, specialization)
    #
    # def load_rt_miss_shader(self, path: str, *specialization, main_function='main'):
    #     return self.load_shader(vk.VK_SHADER_STAGE_MISS_BIT_KHR, path, main_function, specialization)

    def create_rt_hit_group(self, closest_hit: int = None, any_hit: int = None, intersection: int = None):
        return self.w_pipeline.create_hit_group(closest_hit, any_hit, intersection)

    def create_rt_gen_group(self, generation_shader_index: int):
        return self.w_pipeline.create_general_group(generation_shader_index)

    def create_rt_miss_group(self, miss_shader_index: int):
        return self.w_pipeline.create_general_group(miss_shader_index)

    def _get_aligned_size(self, size, align):
        return (size + align - 1) & (~(align - 1))

    def create_rt_program(self, max_miss_shader=10, max_hit_groups=1000):
        shaderHandlerSize = self.w_pipeline.w_device.raytracing_properties.shaderGroupHandleSize
        groupAlignment = self.w_pipeline.w_device.raytracing_properties.shaderGroupBaseAlignment

        raygen_size = self._get_aligned_size(shaderHandlerSize, groupAlignment)
        raymiss_size = self._get_aligned_size(shaderHandlerSize * max_miss_shader, groupAlignment)
        rayhit_size = self._get_aligned_size(shaderHandlerSize * max_hit_groups, groupAlignment)

        w_buffer = self.w_pipeline.w_device.create_buffer(
            raygen_size + raymiss_size + rayhit_size,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            properties=MemoryLocation.CPU)
        return RTProgram(self, w_buffer, raygen_size, raygen_size + raymiss_size)


class CommandManager:

    def __init__(self, device, w_cmdList: vkw.CommandBufferWrapper):
        self.w_cmdList = w_cmdList
        self.device = device

    def __del__(self):
        self.w_cmdList = None
        self.device = None

    @classmethod
    def get_queue_required(cls) -> int:
        pass

    def freeze(self):
        self.w_cmdList.freeze()

    def is_frozen(self):
        return self.w_cmdList.is_frozen()

    def is_closed(self):
        return self.w_cmdList.is_closed()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.device.safe_dispatch_function(lambda: self.w_cmdList.flush_and_wait())


class CopyManager(CommandManager):
    def __init__(self, device, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(device, w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.COPY

    def copy(self, src_resource, dst_resource):
        if src_resource is None or dst_resource is None:
            return
        self.w_cmdList.copy(src_resource.w_resource, dst_resource.w_resource)

    # def copy_image(self, src_image: Image, dst_image: Image):
    #     self.w_cmdList.copy_image(src_image.w_resource, dst_image.w_resource)
    #
    # def copy_buffer_to_image(self, src_buffer: Buffer, dst_image: Image):
    #     self.w_cmdList.copy_buffer_to_image(src_buffer.w_resource, dst_image.w_resource)


class ComputeManager(CopyManager):
    def __init__(self, device, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(device, w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.COMPUTE

    def clear_color(self, image: Image, color):
        self.w_cmdList.clear_color(image.w_resource, color)

    def clear_buffer(self, buffer: Buffer, value: int = 0):
        self.w_cmdList.clear_buffer(buffer.w_resource, value)

    def set_pipeline(self, pipeline: Pipeline):
        if not pipeline.is_closed():
            raise Exception("Error, can not set a pipeline has not been closed.")
        self.w_cmdList.set_pipeline(pipeline=pipeline.w_pipeline)

    def update_sets(self, *sets):
        for s in sets:
            self.w_cmdList.update_bindings_level(s)

    def update_constants(self, stages, **fields):
        self.w_cmdList.update_constants(stages, **{
            f: Interop.to_bytes(t=type(v), value=v)
            for f, v in fields.items()
        })

    def dispatch_groups(self, groups_x: int, groups_y: int = 1, groups_z: int = 1):
        self.w_cmdList.dispatch_groups(groups_x, groups_y, groups_z)

    def dispatch_threads_1D(self, dim_x: int, group_size_x: int = 1024):
        self.dispatch_groups(math.ceil(dim_x / group_size_x))

    def dispatch_threads_2D(self, dim_x: int, dim_y: int, group_size_x: int = 32, group_size_y: int = 32):
        self.dispatch_groups(math.ceil(dim_x / group_size_x), math.ceil(dim_y / group_size_y))

    def dispatch_rays(self, program: RTProgram, dim_x: int, dim_y: int, dim_z: int = 1):
        self.w_cmdList.dispatch_rays(
            program.raygen_slice, program.miss_slice, program.hitgroup_slice,
            dim_x, dim_y, dim_z
        )


class GraphicsManager(ComputeManager):
    def __init__(self, device, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(device, w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.GRAPHICS

    def blit_image(self, src_image: Image, dst_image: Image, filter: Filter = Filter.POINT):
        self.w_cmdList.blit_image(src_image.w_resource, dst_image.w_resource, filter)


class RaytracingManager(GraphicsManager):
    def __init__(self, device, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(device, w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.RAYTRACING

    def build_ads(self, ads: ADS, scratch_buffer: Buffer):
        self.w_cmdList.build_ads(
            ads.w_resource,
            ads.ads_info,
            ads.ranges,
            scratch_buffer.w_resource)


class DeviceManager:

    def __init__(self):
        self.w_state = None
        self.width = 0
        self.height = 0
        self.__copying_on_the_gpu = None
        self.__queue = None
        self.__loop_process = None
        self.__allow_cross_thread_without_looping = False

    def allow_cross_threading(self):
        self.__allow_cross_thread_without_looping = True
        print(
            "[WARNING] Allow access from other threads is dangerous. Think in wrapping the concurrent process in a loop")

    def safe_dispatch_function(self, function):
        if threading.current_thread() == threading.main_thread() or self.__allow_cross_thread_without_looping:
            function()
            return
        if self.__queue is None:
            raise Exception("Can not dispatch cross-thread function without looping")
        event = threading.Event()
        self.__queue.append((function, event))
        event.wait()

    def loop(self, process):
        assert threading.current_thread() == threading.main_thread(), "Loops can only be invoked from main thread!"
        window: Window = self.w_device.get_window()
        if self.__loop_process is not None:
            self.__loop_process.join()  # wait for an unfinished loop first
        self.__queue = []  # start dispatch queue
        self.__loop_process = threading.Thread(target=process)
        self.__loop_process.start()
        while self.__loop_process.is_alive():
            if window is not None:
                ev, args = window.poll_events()
                if ev == Event.CLOSED:
                    return
            while len(self.__queue) > 0:  # consume all pending dispatched functions
                f, event = self.__queue.pop(0)
                f()
                event.set()
        self.__queue = None

    def release(self):
        self.__copying_on_the_gpu = None
        if self.w_device is not None:
            self.w_device.release()
        self.w_device = None

    def __del__(self):
        self.release()

    def __bind__(self, w_device: vkw.DeviceWrapper):
        self.w_device = w_device
        if self.w_device.mode != 0:
            self.width = w_device.get_render_target(0).resource_data.vk_description.extent.width
            self.height = w_device.get_render_target(0).resource_data.vk_description.extent.height

    def render_target(self):
        assert self.w_device.mode != 0, "Can not retrieve a render target of a present-less device."
        return Image(self, self.w_device.get_render_target(self.w_device.get_render_target_index()))

    def presented_render_target(self):
        assert self.w_device.mode != 0, "Can not retrieve a render target of a present-less device."
        return Image(self, self.w_device.get_render_target(
            (self.w_device.get_render_target_index() + self.w_device.get_number_of_frames() - 1)%
            self.w_device.get_number_of_frames()
        ))

    def load_technique(self, technique):
        technique.__bind__(self.w_device)
        technique.__setup__()
        return technique

    def dispatch_technique(self, technique):
        assert technique.w_device, "Technique is not bound to a device, you must load the technique in some point before dispatching"
        technique.__dispatch__()

    def set_debug_name(self, name: str):
        self.w_device.set_debug_name(name)

    def create_buffer(self, size: int, usage: int, memory: MemoryLocation):
        return Buffer(self, self.w_device.create_buffer(size, usage, memory))

    def create_staging(self, image: Image, memory: MemoryLocation = MemoryLocation.CPU):
        size = image.w_resource.get_size()
        return self.create_buffer(size, BufferUsage.TRANSFER_SRC | BufferUsage.TRANSFER_DST, memory).reshape_as(image)

    def create_uniform_buffer(self, usage: int = BufferUsage.UNIFORM,
                              memory: MemoryLocation = MemoryLocation.CPU,
                              **fields) -> Uniform:
        """
        Creates a uniform buffer. By default is created cpu-visible, meaning the updates to the buffer from the cpu
        is efficient, but reading from the gpu might be slower.
        Uniform buffers can access data via fields.
        """
        layout, size = Uniform.process_layout(fields)
        resource = self.w_device.create_buffer(size, usage, memory)
        return Uniform(self, resource, layout)

    def create_structured_buffer(self, count: int,
                                 usage: int = BufferUsage.VERTEX | BufferUsage.TRANSFER_DST,
                                 memory: MemoryLocation = MemoryLocation.GPU,
                                 **fields) -> StructuredBuffer:
        """
        Creates a structured buffer.
        """
        layout, size = Uniform.process_layout(fields)
        resource = self.w_device.create_buffer(size * count, usage, memory)
        return StructuredBuffer(self, resource, layout, size)

    def create_indices_buffer(self, count: int,
                              usage: int = BufferUsage.INDEX | BufferUsage.TRANSFER_DST,
                              memory: MemoryLocation = MemoryLocation.GPU):
        return IndexBuffer(self, self.w_device.create_buffer(count * 4, usage, memory))

    def create_image(self, image_type: ImageType, is_cube: bool, image_format: Format,
                     width: int, height: int, depth: int,
                     mips: int, layers: int,
                     usage: int, memory: MemoryLocation):
        # TODO: PROPERLY CHECK FOR OPTIMAL TILING SUPPORT
        linear = False  # image_format == Format.VEC3 or bool(usage & (vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        return Image(self, self.w_device.create_image(
            image_type, image_format, vk.VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT if is_cube else 0,
            vk.VkExtent3D(width, height, depth), mips, layers, linear, layout, usage, memory
        ))

    def create_gpu_wrapped_ptr(self):
        resource = self.w_device.create_buffer(8, BufferUsage.UNIFORM, MemoryLocation.CPU)
        return WrappedPtr(self, resource)

    def create_gpu_wrapped_ptr_collection(self, count):
        resource = self.w_device.create_buffer(8 * count, BufferUsage.UNIFORM, MemoryLocation.CPU)
        return WrappedPtrCollection(self, resource, count)

    def create_triangle_collection(self) -> TriangleCollection:
        return TriangleCollection(device=self.w_device)

    def create_geometry_ads(self, collection: GeometryCollection) -> ADS:
        ads, info, ranges, handle, scratch_size = self.w_device.create_ads(
            geometry_type=collection.get_collection_type(),
            descriptions=[
                (v.w_resource, v.stride, None if i is None else i.w_resource, None if t is None else t.w_resource)
                for v, i, t in collection.descriptions
            ]
        )
        return ADS(self, ads, handle, scratch_size, info, ranges)

    def create_scene_ads(self, instance_buffer: InstanceBuffer):
        ads, info, ranges, handle, scratch_size = self.w_device.create_ads(
            geometry_type=vk.VK_GEOMETRY_TYPE_INSTANCES_KHR,
            descriptions=[
                instance_buffer.w_resource
            ]
        )
        return ADS(self, ads, handle, scratch_size, info, ranges, instance_buffer)

    def create_instance_buffer(self, instances: int, memory: MemoryLocation = MemoryLocation.GPU):
        buffer = self.w_device.create_buffer(instances * 64,
                                             BufferUsage.RAYTRACING_ADS_READ | BufferUsage.TRANSFER_DST,
                                             memory)
        return InstanceBuffer(self, buffer, instances)

    def create_sampler(self,
                       mag_filter: Filter = Filter.POINT,
                       min_filter: Filter = Filter.POINT,
                       mipmap_mode: MipMapMode = MipMapMode.POINT,
                       address_U: AddressMode = AddressMode.REPEAT,
                       address_V: AddressMode = AddressMode.REPEAT,
                       address_W: AddressMode = AddressMode.REPEAT,
                       mip_LOD_bias: float = 0.0,
                       enable_anisotropy: bool = False,
                       max_anisotropy: float = 0.0,
                       enable_compare: bool = False,
                       compare_op: CompareOp = CompareOp.NEVER,
                       min_LOD: float = 0.0,
                       max_LOD: float = 0.0,
                       border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
                       use_unnormalized_coordinates: bool = False
                       ) -> Sampler:
        return self.w_device.create_sampler(mag_filter, min_filter, mipmap_mode,
                                            address_U, address_V, address_W,
                                            mip_LOD_bias, 1 if enable_anisotropy else 0,
                                            max_anisotropy, 1 if enable_compare else 0,
                                            compare_op, min_LOD, max_LOD,
                                            border_color,
                                            1 if use_unnormalized_coordinates else 0)

    def create_compute_pipeline(self):
        return Pipeline(self.w_device.create_pipeline(
            vk_pipeline_type=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO))

    def create_graphics_pipeline(self):
        return Pipeline(self.w_device.create_pipeline(
            vk_pipeline_type=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO))

    def create_raytracing_pipeline(self):
        return Pipeline(self.w_device.create_pipeline(
            vk_pipeline_type=vk.VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR))

    def _get_queue_manager(self, queue_bits: int):
        return self.w_device.create_cmdList(queue_bits)

    def get_graphics(self) -> GraphicsManager:
        return GraphicsManager(self, self._get_queue_manager(QueueType.GRAPHICS))

    def get_compute(self) -> ComputeManager:
        return ComputeManager(self, self._get_queue_manager(QueueType.COMPUTE))

    def get_raytracing(self) -> RaytracingManager:
        return RaytracingManager(self, self._get_queue_manager(QueueType.RAYTRACING))

    def get_copy(self) -> CopyManager:
        return CopyManager(self, self._get_queue_manager(QueueType.COPY))

    def submit(self, man: CommandManager):
        assert man.is_frozen(), "Only frozen managers can be submitted"
        self.safe_dispatch_function(lambda: man.w_cmdList.flush_and_wait())

    def flush(self):
        self.safe_dispatch_function(lambda: self.w_device.flush_pending_and_wait())

    def vulkanized(self, obj):
        '''
        Moves the memory of the tensor or models to vulkan memory.
        '''
        if hasattr(obj, 'is_vulkanized') and obj.is_vulkanized:
            return obj
        if isinstance(obj, torch.Tensor):
            assert obj.device == torch.device('cuda:0'), "Only tensors in same device than vulkan can be wrap"
            t: torch.Tensor = obj
            assert t.is_contiguous(), "Can not map to buffer no-contiguous tensors"
            base_tensor = t if t._base is None else t._base
            if hasattr(base_tensor, "is_vulkanized") and obj.is_vulkanized:
                return t  # a subset of a vulkan memory is in vulkan memory.
            st = t.storage()
            # Create buffer with vulkan memory to backup
            vk_back_buffer = self.create_tensor(*base_tensor.shape, dtype=base_tensor.dtype)
            # Compute the size of vulkan replacement
            size = st.element_size() * st.size()
            # Copy current tensor data to vulkan memory
            cuda.cuMemcpy(vk_back_buffer.cuda_ptr(), st.data_ptr(), size)
            # Get a new storage with the new data
            b_t = vk_back_buffer.as_tensor()
            base_tensor.data = b_t
            base_tensor._back_buffer = vk_back_buffer
            # Tag the base tensor as vulkanized to avoid further computations
            base_tensor.is_vulkanized = True
            # Wish for the best
            return t
        if isinstance(obj, torch.nn.Module):
            model: torch.nn.Module = obj
            state_dic = model.state_dict()
            for k in state_dic:
                state_dic[k] = self.vulkanized(state_dic[k])
            model.load_state_dict(state_dic)
            model.is_vulkanized = True
            return model
        raise Exception('Can not move the object to vulkan memory.')

    def as_buffer(self, obj):
        if isinstance(obj, Buffer):
            return obj
        if isinstance(obj, torch.Tensor):
            t: torch.Tensor = obj
            base_tensor = t if t._base is None else t._base
            if not (hasattr(base_tensor, 'is_vulkanized') and base_tensor.is_vulkanized):
                self.vulkanized(base_tensor)
            # assert hasattr(base_tensor, 'is_vulkanized') and base_tensor.is_vulkanized, "Only vulkanized tensors can be treated as buffers"
            back_buffer = base_tensor._back_buffer
            size_of_t = t.nelement() * t.element_size()
            offset_of_t = t.storage_offset() * t.element_size()
            b = back_buffer.slice(offset_of_t, size_of_t)
            b._info['shape'] = [t.shape]
            b._info['dtype'] = t.dtype
            return b
        raise Exception('Can treat as buffer this kind of object')

    def create_tensor(self, *shape, dtype = None, memory = MemoryLocation.GPU, clear = True) -> Buffer:
        assert all(isinstance(i, int) for i in shape), "All shape dimension is int"
        if dtype is None:
            dtype = torch.float32
        element_size = Interop.size_of_dtype(dtype)
        numel = math.prod(shape)
        size = element_size * numel
        b = self.create_buffer(size, BufferUsage.TENSOR, memory)
        b._info['shape'] = [shape]
        b._info['dtype'] = dtype
        if clear:
            cuda.cuMemsetD8(b.cuda_ptr(), 0, b.size)
        return b

    def create_tensor_like(self, t, memory = None, clear: bool = True):
        if isinstance(t, torch.Tensor):
            shape = t.shape
            dtype = t.dtype
            if memory is None:
                memory = MemoryLocation.CPU if t.device == torch.device('cpu') else MemoryLocation.GPU
        elif isinstance(t, Resource):
            shape = t.get_shape()
            assert len(shape) == 1, "Can only create buffer-tensors from single resource shapes"
            shape = shape[0]
            dtype = t.get_dtype()
            if memory is None:
                memory = MemoryLocation.GPU if t.w_resource.resource_data.is_gpu else MemoryLocation.CPU
        else:
            raise Exception('Unsupported type to get shape and dtype')
        return self.create_tensor(*shape, dtype=dtype, memory=memory, clear=clear)

    def __enter__(self):
        self.__previous_device = __ACTIVE_DEVICE__
        return device_manager(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        device_manager(self.__previous_device)


class Presenter(DeviceManager):

    def __init__(self):
        super().__init__()

    def begin_frame(self):
        self.safe_dispatch_function(lambda: self.w_device.begin_frame())

    def end_frame(self):
        self.safe_dispatch_function(lambda: self.w_device.end_frame())

    def get_window(self):
        return self.w_device.get_window()


__ACTIVE_DEVICE__: DeviceManager = None


def device_manager(new_device: DeviceManager = None) -> DeviceManager:
    global __ACTIVE_DEVICE__
    if new_device is not None:
        __ACTIVE_DEVICE__ = new_device
    return __ACTIVE_DEVICE__


def create_presenter(width: int, height: int,
                     format: Format,
                     mode: PresenterMode,
                     usage: int = ImageUsage.RENDER_TARGET,
                     device: int = 0,
                     debug: bool = False,
                     set_active: bool = True) -> Presenter:
    """
    Creates a device manager with capacity for presentation. This is the core of vulkan graphics call.
    This method automatically sets created device as active, further actions will use it.
    To change to other devices use device method. e.g: device_manager(other_device)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    state = vkw.DeviceWrapper(
        width=width,
        height=height,
        format=format,
        mode=mode,
        render_usage=usage,
        enable_validation_layers=debug
    )
    presenter = Presenter()
    presenter.__bind__(state)
    return device_manager(presenter) if set_active else presenter


class _Frame:
    def __enter__(self):
        p : Presenter = __ACTIVE_DEVICE__
        p.begin_frame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        p : Presenter = __ACTIVE_DEVICE__
        p.end_frame()


def create_device(*, device: int = 0, debug: bool = False, set_active: bool = True) -> DeviceManager:
    """
    Creates a device manager. This is the core of vulkan graphics call.
    This method automatically sets created device as active, further actions will use it.
    To change to other devices use device method. e.g: device_manager(other_device)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    state = vkw.DeviceWrapper(
        width=0,
        height=0,
        format=Format.NONE,
        mode=PresenterMode.NONE,
        render_usage=ImageUsage.NONE,
        enable_validation_layers=debug
    )
    dev = DeviceManager()
    dev.__bind__(state)
    return device_manager(dev) if set_active else dev


def quit():
    """
    Releases the active device from future usages.
    Use this function at the end of the program execution to avoid any hanging resources
    and properly shutdown vulkan objects
    """
    global __ACTIVE_DEVICE__
    if __ACTIVE_DEVICE__ is not None:
        __ACTIVE_DEVICE__.release()
    __ACTIVE_DEVICE__ = None


class Technique(DeviceManager):
    def __setup__(self):
        pass

    def __dispatch__(self):
        pass


def Extends(class_):
    def wrapper(function):
        setattr(class_, function.__name__, function)

    return wrapper


def _check_active_device(f):
    def wrap(*args, **kwargs):
        assert __ACTIVE_DEVICE__ is not None, "Create a device, a presenter or set active a device."
        return f(*args, **kwargs)

    return wrap


@_check_active_device
def frame():
    assert __ACTIVE_DEVICE__.w_device.mode > 0, "Can only use frames if device is a presenter. Use create_presenter instead of create_device"
    return _Frame()


@_check_active_device
def tensor_like(t: typing.Union[torch.Tensor, Resource], memory: MemoryLocation.GPU = None, clear: bool = True) -> Buffer:
    """
    Creates a buffer using a tensor as reference. The shape and element type are saved internally for
    future conversion of the buffer to torch, cupy and numpy tensors.
    The buffer is created with possible usage to transfer to/from, storage binding and gpu addressing.
    """
    return __ACTIVE_DEVICE__.create_tensor_like(t, memory, clear)


@_check_active_device
def staging(image: Image, memory: MemoryLocation = MemoryLocation.CPU):
    """
    Creates a buffer with sufficient size to represent all image subresources linearly.
    The buffer has already internally the shape of this image and tensors and arrays are constructed
    using this partition.
    e.g.
    i = image(...)
    s = staging(i)  # By default creates on CPU memory.
    s.as_numpy()[0][:,:] = ... # copies values to the 0 subresource
    i.load(s)  # copy staging buffer into image
    """
    return __ACTIVE_DEVICE__.create_staging(image, memory)


@_check_active_device
def tensor(*shape, dtype: dtype = None, memory:MemoryLocation = MemoryLocation.GPU, clear: bool = True) -> Buffer:
    """
    Creates a buffer using a specific shape and type for future conversion to torch, cupy and numpy tensors.
    The buffer is created with possible usage to transfer to/from, storage binding and gpu addressing.
    """
    return __ACTIVE_DEVICE__.create_tensor(*shape, dtype=dtype, memory=memory, clear=clear)


@_check_active_device
def pipeline_raytracing() -> Pipeline:
    """
    Creates a Pipeline to manage the setup of a raytracing pipeline, resources, shaders and other attributes.
    """
    return __ACTIVE_DEVICE__.create_raytracing_pipeline()


@_check_active_device
def pipeline_graphics() -> Pipeline:
    """
    Creates a Pipeline to manage the setup of a graphics pipeline, resources, shaders and other attributes.
    """
    return __ACTIVE_DEVICE__.create_graphics_pipeline()


@_check_active_device
def pipeline_compute() -> Pipeline:
    """
    Creates a Pipeline to manage the setup of a compute pipeline, resources, shaders and other attributes.
    """
    return __ACTIVE_DEVICE__.create_compute_pipeline()


@_check_active_device
def sampler(mag_filter: Filter = Filter.POINT,
            min_filter: Filter = Filter.POINT,
            mipmap_mode: MipMapMode = MipMapMode.POINT,
            address_U: AddressMode = AddressMode.REPEAT,
            address_V: AddressMode = AddressMode.REPEAT,
            address_W: AddressMode = AddressMode.REPEAT,
            mip_LOD_bias: float = 0.0,
            enable_anisotropy: bool = False,
            max_anisotropy: float = 0.0,
            enable_compare: bool = False,
            compare_op: CompareOp = CompareOp.NEVER,
            min_LOD: float = 0.0,
            max_LOD: float = 0.0,
            border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
            use_unnormalized_coordinates: bool = False
            ) -> Sampler:
    """
    Creates a sampler that can be used to bind texture objects.
    """
    return __ACTIVE_DEVICE__.create_sampler(
        mag_filter, min_filter, mipmap_mode, address_U, address_V, address_W, mip_LOD_bias, enable_anisotropy,
        max_anisotropy,
        enable_compare, compare_op, min_LOD, max_LOD, border_color, use_unnormalized_coordinates
    )


@_check_active_device
def sampler_linear(address_U: AddressMode = AddressMode.REPEAT,
                  address_V: AddressMode = AddressMode.REPEAT,
                  address_W: AddressMode = AddressMode.REPEAT,
                  mip_LOD_bias: float = 0.0,
                  enable_anisotropy: bool = False,
                  max_anisotropy: float = 0.0,
                  enable_compare: bool = False,
                  compare_op: CompareOp = CompareOp.NEVER,
                  min_LOD: float = 0.0,
                  max_LOD: float = 1000.0,
                  border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
                  use_unnormalized_coordinates: bool = False
                  ) -> Sampler:
    """
    Creates a linear sampler that can be used to bind texture objects.
    """
    return __ACTIVE_DEVICE__.create_sampler(
        Filter.LINEAR, Filter.LINEAR, MipMapMode.LINEAR, address_U, address_V, address_W, mip_LOD_bias, enable_anisotropy,
        max_anisotropy,
        enable_compare, compare_op, min_LOD, max_LOD, border_color, use_unnormalized_coordinates
    )


@_check_active_device
def image_1D(image_format: Format, width: int, mips=None, layers=1,
                     usage: int = ImageUsage.TRANSFER_DST | ImageUsage.SAMPLED) -> Image:
    """
    Creates a one-dimensional image object on the GPU. If mips is None, then the maximum possible value is used.
    """
    if mips is None:
        mips = int(math.log(width, 2)) + 1
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_1D, False, image_format,
                             width, 1, 1, mips, layers, usage, MemoryLocation.GPU)


@_check_active_device
def image_2D(image_format: Format, width: int, height: int, mips=None, layers=1,
                     usage: int = ImageUsage.TRANSFER_DST | ImageUsage.SAMPLED) -> Image:
    """
    Creates a two-dimensional image object on the GPU. If mips is None, then the maximum possible value is used.
    """
    if mips is None:
        mips = int(math.log(max(width, height), 2)) + 1
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_2D, False, image_format,
                             width, height, 1, mips, layers, usage, MemoryLocation.GPU)


@_check_active_device
def image_3D(image_format: Format, width: int, height: int, depth: int, mips : int = None, layers : int = 1,
                     usage: int = ImageUsage.TRANSFER_DST | ImageUsage.SAMPLED) -> Image:
    """
    Creates a three-dimensional image object on the GPU. If mips is None, then the maximum possible value is used.
    """
    if mips is None:
        mips = int(math.log(max(width, height, depth), 2)) + 1
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_3D, False, image_format,
                             width, height, depth, mips, layers, usage, MemoryLocation.GPU)


@_check_active_device
def image(image_type: ImageType, is_cube: bool, image_format: Format,
                 width: int, height: int, depth: int,
                 mips: int, layers: int,
                 usage: int, memory: MemoryLocation) -> Image:
    """
    Creates an image object on the specified memory (HOST or DEVICE).
    """
    return __ACTIVE_DEVICE__.create_image(image_type, is_cube, image_format,
                              width, height, depth, mips, layers, usage, memory)


@_check_active_device
def render_target(image_format: Format, width: int, height: int) -> Image:
    """
    Creates a two-dimensional image object on the GPU to be used as render target.
    """
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_2D, False, image_format,
                             width, height, 1, 1, 1, ImageUsage.RENDER_TARGET, MemoryLocation.GPU)


@_check_active_device
def depth_stencil(image_format: Format, width: int, height: int) -> Image:
    """
    Creates a two-dimensional image object on the GPU to be used as depth stencil buffer.
    """
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_2D, False, image_format,
                             width, height, 1, 1, 1, ImageUsage.DEPTH_STENCIL, MemoryLocation.GPU)


@_check_active_device
def scratch_buffer(*adss) -> Buffer:
    """
    Creates a buffer on the GPU to be used as scratch buffer for acceleration-datastructure creation.
    """
    size = max(a.scratch_size for a in adss)
    return __ACTIVE_DEVICE__.create_buffer(size, BufferUsage.STORAGE | BufferUsage.GPU_ADDRESS, MemoryLocation.GPU)


@_check_active_device
def instance_buffer(instances: int, memory: MemoryLocation = MemoryLocation.GPU) -> InstanceBuffer:
    """
    Creates a buffer on the GPU to be used to store instances of a scene acceleration-datastructure.
    """
    return __ACTIVE_DEVICE__.create_instance_buffer(instances, memory)


@_check_active_device
def ads_scene(instance_buffer: InstanceBuffer) -> ADS:
    """
    Creates an acceleration data structure for the scene elements (top-level ads).
    """
    return __ACTIVE_DEVICE__.create_scene_ads(instance_buffer)


@_check_active_device
def ads_model(collection: GeometryCollection) -> ADS:
    """
    Creates an acceleration data structure for a model formed by a set of geometries (bottom-level ads).
    """
    return __ACTIVE_DEVICE__.create_geometry_ads(collection)


@_check_active_device
def buffer(size: int, usage: int, memory: MemoryLocation) -> Buffer:
    """
    Creates a buffer for a generic usage. Cuda-visible buffers exposes a cuda_ptr and
    can be wrap as tensors zero-copy operation.
    """
    return __ACTIVE_DEVICE__.create_buffer(size, usage, memory)


@_check_active_device
def uniform_buffer(usage: int = BufferUsage.UNIFORM | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                          memory: MemoryLocation = MemoryLocation.CPU,
                          **fields) -> Uniform:
    """
    Creates a buffer for a uniform store. Uniform CPU data can be updated (cpu version) accessing to the fields.
    To finally update the resource (in case is allocated on the gpu) use flush_cpu().
    """
    return __ACTIVE_DEVICE__.create_uniform_buffer(usage, memory, **fields)


@_check_active_device
def structured_buffer(count: int,
                             usage: int = BufferUsage.VERTEX | BufferUsage.TRANSFER_DST,
                             memory: MemoryLocation = MemoryLocation.GPU,
                             **fields):
    """
    Creates a buffer for a structured store. Each index is a Uniform that
    can be updated (cpu version) accessing to the fields.
    To finally update the resource (in case is allocated on the gpu) use flush_cpu().
    """
    return __ACTIVE_DEVICE__.create_structured_buffer(count, usage, memory, **fields)


@_check_active_device
def index_buffer(count: int,
                          usage: int = BufferUsage.INDEX | BufferUsage.TRANSFER_DST,
                          memory: MemoryLocation = MemoryLocation.GPU):
    """
    Creates a buffer for an indices store. Each index is a int32 value that
    can be updated (cpu version).
    To finally update the resource (in case is allocated on the gpu) use flush_cpu().
    """
    return __ACTIVE_DEVICE__.create_indices_buffer(count, usage, memory)


@_check_active_device
def wrapped_ptr() -> WrappedPtr:
    """
    Creates an uniform buffer for an int64 ptr store.
    Use wrap method to take the ptr from a valid vulkan-compatible object (vulkanized tensors, buffers) or
    set directly the ptr field (in that case flush_cpu is necessary).
    """
    return __ACTIVE_DEVICE__.create_gpu_wrapped_ptr()


def wrapped_ptr_collection(count: int) -> WrappedPtrCollection:
    """
    Creates a buffer to store a collection of int64.
    Use wrap method to take the ptr from a valid vulkan-compatible objects (vulkanized tensors, buffers).
    """
    return __ACTIVE_DEVICE__.create_gpu_wrapped_ptr_collection(count)


@_check_active_device
def triangle_collection() -> TriangleCollection:
    return __ACTIVE_DEVICE__.create_triangle_collection()


@_check_active_device
def to_vulkan(obj: typing.Any) -> typing.Any:
    """
    Prepares the object to make the memory directly accessible from vulkan.
    Supports tensors. The storage of the tensor is changed to use a vulkan memory by a single-time transfer operation.
    After that, all usages of the tensor as a buffer is a Zero-copy operation.
    """
    return __ACTIVE_DEVICE__.vulkanized(obj)


@_check_active_device
def as_buffer(obj: typing.Any) -> Buffer:
    """
    Treats the object as a vulkan buffer if possible.
    """
    return __ACTIVE_DEVICE__.as_buffer(obj)


@_check_active_device
def flush():
    """
    Flushes all pending work of vulkan submissions and wait for completion.
    """
    __ACTIVE_DEVICE__.flush()


def flush_cuda():
    """
    Waits for cuda streams to finish before using the buffers in vulkan.
    """
    torch.cuda.synchronize()


@_check_active_device
def compute_manager() -> ComputeManager:
    """
    Gets a compute manager object that can be used to populate with compute commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_compute()


@_check_active_device
def copy_manager() -> CopyManager:
    """
    Gets a copy manager object that can be used to populate with transfer commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_copy()


@_check_active_device
def graphics_manager() -> GraphicsManager:
    """
    Gets a graphics manager object that can be used to populate with graphics commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_graphics()


@_check_active_device
def raytracing_manager() -> RaytracingManager:
    """
    Gets a raytracing manager object that can be used to populate with raytracing commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_raytracing()


@_check_active_device
def submit(man: CommandManager):
    """
    Allows to submit the command list save in a manager using freeze.
    """
    __ACTIVE_DEVICE__.submit(man)


@_check_active_device
def current_render_target() -> Image:
    """
    Gets the current render target. This function fails if the active device has not presenter-capabilities.
    """
    return __ACTIVE_DEVICE__.render_target()


@_check_active_device
def presented_render_target() -> Image:
    """
    Gets the last presented render target. This function fails if the active device has not presenter-capabilities.
    """
    return __ACTIVE_DEVICE__.presented_render_target()


@_check_active_device
def load_technique(technique):
    return __ACTIVE_DEVICE__.load_technique(technique)


@_check_active_device
def dispatch_technique(technique):
    return __ACTIVE_DEVICE__.dispatch_technique(technique)


@_check_active_device
def loop(process):
    """
    Creates a thread to execute the process safety dispatching vulkan calls in the main thread
    """
    __ACTIVE_DEVICE__.loop(process)


@_check_active_device
def allow_cross_threading():
    """
    Allows dispatching cross-threading vulkan calls. The safest way is to include the cross-threading code
    inside a loop function
    """
    __ACTIVE_DEVICE__.allow_cross_threading()


@_check_active_device
def set_debug_name(name: str):
    __ACTIVE_DEVICE__.set_debug_name(name)