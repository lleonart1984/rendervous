"""
Vulkan wrappers to represent:
Device (manage all internal objects for instance, device, swapchain, queues, pools, descriptors)
Resource (manage all internal objects for a resource, memory, staging memory)
CommandPool (creates and flushes commands)
CommandBuffer (represents a command list to be populated)
GPUTask (synchronization object to wait for GPU completation)
"""
from typing import *
from vulkan import *
import ctypes
from enum import Enum
import numpy as np
import torch
import struct
import os
from cuda import cuda as cuda
if os.name == 'nt':
    import win32api
import gc
import weakref
from ._allocator import VulkanMemoryManager, VulkanMemory


__TRACE__ = False


def trace_destroying(function):
    def wrapper(self, *args):
        if __TRACE__:
            print(f"[INFO] Destroying {type(self)}...", end='')
        function(self, *args)
        if __TRACE__:
            print("done.")
    return wrapper


class ResourceState:
    def __init__(self, vk_access, vk_stage, vk_layout, queue_index):
        self.vk_access = vk_access
        self.vk_stage = vk_stage
        self.vk_layout = vk_layout
        self.queue_index = queue_index

    def __iter__(self):
        yield self.vk_access
        yield self.vk_stage
        yield self.vk_layout
        yield self.queue_index


_FORMAT_DESCRIPTION = {
    VK_FORMAT_R32_SFLOAT: (1, 'f'),
    VK_FORMAT_R32_SINT: (1, 'i'),
    VK_FORMAT_R32_UINT: (1, 'u'),

    VK_FORMAT_R32G32_SFLOAT: (2, 'f'),
    VK_FORMAT_R32G32_SINT: (2, 'i'),
    VK_FORMAT_R32G32_UINT: (2, 'u'),

    VK_FORMAT_R32G32B32_SFLOAT: (3, 'f'),
    VK_FORMAT_R32G32B32_SINT: (3, 'i'),
    VK_FORMAT_R32G32B32_UINT: (3, 'u'),

    VK_FORMAT_R32G32B32A32_SFLOAT: (4, 'f'),
    VK_FORMAT_R32G32B32A32_SINT: (4, 'i'),
    VK_FORMAT_R32G32B32A32_UINT: (4, 'u'),

    VK_FORMAT_R8G8B8A8_UNORM: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SNORM: (4, 'b'),
    VK_FORMAT_R8G8B8A8_USCALED: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SSCALED: (4, 'b'),
    VK_FORMAT_R8G8B8A8_UINT: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SINT: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SRGB: (4, 'b'),

    VK_FORMAT_B8G8R8A8_UNORM: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SNORM: (4, 'b'),
    VK_FORMAT_B8G8R8A8_USCALED: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SSCALED: (4, 'b'),
    VK_FORMAT_B8G8R8A8_UINT: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SINT: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SRGB: (4, 'b'),
}


_FORMAT_SIZES = {

    VK_FORMAT_R32_SFLOAT: 4,
    VK_FORMAT_R32_SINT: 4,
    VK_FORMAT_R32_UINT: 4,

    VK_FORMAT_R32G32_SFLOAT: 8,
    VK_FORMAT_R32G32_SINT: 8,
    VK_FORMAT_R32G32_UINT: 8,

    VK_FORMAT_R32G32B32_SFLOAT: 12,
    VK_FORMAT_R32G32B32_SINT: 12,
    VK_FORMAT_R32G32B32_UINT: 12,

    VK_FORMAT_R32G32B32A32_SFLOAT: 16,
    VK_FORMAT_R32G32B32A32_SINT: 16,
    VK_FORMAT_R32G32B32A32_UINT: 16,

    VK_FORMAT_R8G8B8A8_UNORM: 4,
    VK_FORMAT_R8G8B8A8_SNORM: 4,
    VK_FORMAT_R8G8B8A8_USCALED: 4,
    VK_FORMAT_R8G8B8A8_SSCALED: 4,
    VK_FORMAT_R8G8B8A8_UINT: 4,
    VK_FORMAT_R8G8B8A8_SINT: 4,
    VK_FORMAT_R8G8B8A8_SRGB: 4,

    VK_FORMAT_B8G8R8A8_UNORM: 4,
    VK_FORMAT_B8G8R8A8_SNORM: 4,
    VK_FORMAT_B8G8R8A8_USCALED: 4,
    VK_FORMAT_B8G8R8A8_SSCALED: 4,
    VK_FORMAT_B8G8R8A8_UINT: 4,
    VK_FORMAT_B8G8R8A8_SINT: 4,
    VK_FORMAT_B8G8R8A8_SRGB: 4,
}


class SubresourceFootprint:
    def __init__(self, dim: tuple, element_stride, row_pitch, slice_pitch, size):
        self.dim = dim
        self.element_stride = element_stride
        self.row_pitch = row_pitch
        self.slice_pitch = slice_pitch
        self.size = size


class ResourceData:
    def __init__(self,
                 vk_device,
                 vk_description,
                 vk_memory_location,
                 vk_resource,
                 w_memory,
                 is_buffer,
                 initial_state
                 # is_linear = False
                 ):
        self.vk_device = vk_device
        self.vk_description = vk_description
        self.vk_memory_location = vk_memory_location
        self.vk_resource = vk_resource
        self.w_memory : VulkanMemory = w_memory
        self.current_state = initial_state
        self.is_buffer = is_buffer
        self.is_ads = False
        self.ads = None
        self.is_cpu_visible = bool(vk_memory_location & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        self.is_gpu = bool(vk_memory_location & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        # self.is_linear = is_linear  # Gets if this resource is linear and memory lies linearly on the gpu
        if is_buffer:
            self.full_slice = {"offset": 0, "size": vk_description.size}
        else:
            self.full_slice = {
                "mip_start": 0, "mip_count": vk_description.mipLevels,
                "array_start": 0, "array_count": vk_description.arrayLayers
            }
        cpu_footprints = []  # Subresources footprints in a continuous memory
        cpu_offsets = []
        self.cpu_size = 0
        if self.is_buffer:
            self.element_size = 1
            cpu_footprints.append(SubresourceFootprint(
                (vk_description.size, 1, 1), 1, vk_description.size, vk_description.size, vk_description.size))
            cpu_offsets.append(self.cpu_size)
            self.cpu_size = vk_description.size
        else:
            self.element_size = _FORMAT_SIZES[vk_description.format]
            dim = self.vk_description.extent.width, self.vk_description.extent.height, self.vk_description.extent.depth
            for m in range(vk_description.mipLevels):
                size = dim[0] * dim[1] * dim[2] * self.element_size
                for a in range(vk_description.arrayLayers):
                    cpu_footprints.append(SubresourceFootprint(
                        dim, self.element_size, dim[0]*self.element_size, dim[0]*dim[1]*self.element_size, size))
                    cpu_offsets.append(self.cpu_size)
                    self.cpu_size += size
                dim = max(1, dim[0] // 2), max(1, dim[1] // 2), max(1, dim[2] // 2)
        self.cpu_footprints = cpu_footprints
        self.cpu_offsets = cpu_offsets
        ## I have decided not to support linear images. they are incredible restricted in vulkan!
        # if is_buffer:
        #     self.footprints = cpu_footprints
        #     self.offsets = [0]
        # else:
        #     if self.is_linear:
        #         self.footprints = []
        #         self.offsets = []
        #         offset = 0
        #         for m in range(vk_description.mipLevels):
        #             for a in range(vk_description.arrayLayers):
        #                 layout = vkGetImageSubresourceLayout(device.vk_device, vk_resource, VkImageSubresource(
        #                     VK_IMAGE_ASPECT_COLOR_BIT,
        #                     m, a
        #                 ))
        #                 cpu_layout = cpu_footprints[m*vk_description.arrayLayers + a]
        #                 self.footprints.append(SubresourceFootprint(
        #                     dim=cpu_layout.dim,
        #                     element_stride=cpu_layout.element_stride,
        #                     row_pitch=layout.rowPitch,
        #                     slice_pitch=layout.depthPitch,
        #                     size=layout.size
        #                 ))
        #                 self.offsets.append(offset)
        #                 offset += layout.size
        #     else:
        #         # non linear resources can only be treated as a hole
        #         self.footprints = [SubresourceFootprint(
        #         (w_memory.size // self.element_size, 1, 1), self.element_size, w_memory.size, w_memory.size, w_memory.size)]
        #         self.offsets = [0]

    def bind_ads(self, ads):
        self.is_ads = True
        self.ads = ads

    vkDestroyAccelerationStructure = None

    def _vk_destroy(self):
        if self.vk_device is None:
            return
        if self.w_memory is not None:
            if self.is_buffer:
                if self.is_ads:
                    ResourceData.vkDestroyAccelerationStructure(self.vk_device, self.ads, None)
                vkDestroyBuffer(self.vk_device, self.vk_resource, None)
            else:
                vkDestroyImage(self.vk_device, self.vk_resource, None)
            self.w_memory.free()
        self.vk_device = None
        self._staging = None
        self.w_memory = None

    @trace_destroying
    def __del__(self):
        self._vk_destroy()

    def get_device_ptr(self, slice = None):
        assert self.is_buffer, "Only buffers can be considered continuous in memory."
        if slice is None:
            slice = self.full_slice
        start_ptr = self.w_memory.device_ptr()
        return start_ptr + slice['offset']

    def get_cuda_ptr(self, slice = None):
        assert self.is_buffer, "Only buffers can be considered continuous in memory."
        if slice is None:
            slice = self.full_slice
        start_ptr = self.w_memory.cuda_ptr()
        return start_ptr + slice['offset']

    def get_host_ptr(self, slice = None):
        assert self.is_buffer, "Only buffers can be considered continuous in memory."
        if slice is None:
            slice = self.full_slice
        start_ptr = self.w_memory.host_ptr()
        return start_ptr + slice['offset']

    def get_tensor(self, slice = None):
        assert self.is_buffer, "Only buffers can be considered continuous in memory."
        if slice is None:
            slice = self.full_slice
        start, end = slice['offset'], slice['offset'] + slice['size']
        return self.w_memory.get_tensor()[start:end]

    def get_cpu_footprint_and_offset(self, subresource_index):
        return self.cpu_footprints[subresource_index], self.cpu_offsets[subresource_index]

    def get_subresource_index(self, mip: int, arr: int):
        return arr + mip * self.full_slice['array_count']

    def get_subresource_indices(self, slice):
        if self.is_buffer:
            yield 0
        for m in range(slice['mip_start'], slice['mip_start'] + slice['mip_count']):
            for a in range(slice['array_start'], slice['array_start'] + slice['array_count']):
                yield self.get_subresource_index(m, a)

    @staticmethod
    def get_subresources(slice):
        return VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=slice["mip_start"],
            levelCount=slice["mip_count"],
            baseArrayLayer=slice["array_start"],
            layerCount=slice["array_count"]
        )

    @staticmethod
    def get_subresource_layers(slice):
        return VkImageSubresourceLayers(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            mipLevel=slice["mip_start"],
            baseArrayLayer=slice["array_start"],
            layerCount=slice["array_count"]
        )

    def add_barrier(self, vk_cmdList, slice, state: ResourceState):
        srcAccessMask, srcStageMask, oldLayout, srcQueue = self.current_state
        dstAccessMask, dstStageMask, newLayout, dstQueue = state

        srcQueue = VK_QUEUE_FAMILY_IGNORED
        dstQueue = VK_QUEUE_FAMILY_IGNORED

        if self.is_buffer:
            if not self.is_ads:
                barrier = VkBufferMemoryBarrier(
                    buffer=self.vk_resource,
                    srcAccessMask=srcAccessMask,
                    dstAccessMask=dstAccessMask,
                    srcQueueFamilyIndex=srcQueue,
                    dstQueueFamilyIndex=dstQueue,
                    offset=slice["offset"],
                    size=slice["size"]
                )
                vkCmdPipelineBarrier(vk_cmdList, srcStageMask, dstStageMask, 0,
                                     0, None, 1, barrier, 0, None)
        else:
            barrier = VkImageMemoryBarrier(
                image=self.vk_resource,
                srcAccessMask=srcAccessMask,
                dstAccessMask=dstAccessMask,
                oldLayout=oldLayout,
                newLayout=newLayout,
                srcQueueFamilyIndex=srcQueue,
                dstQueueFamilyIndex=dstQueue,
                subresourceRange=ResourceData.get_subresources(slice)
            )
            vkCmdPipelineBarrier(vk_cmdList, srcStageMask, dstStageMask, 0,
                                 0, None, 0, None, 1, barrier)
        self.current_state = state

    @staticmethod
    def copy_from_to(vk_cmdList, src, src_slice, dst, dst_slice):
        def get_buffer_image_regions(buf, buf_slice, img, img_slice):
            regions = []
            buf_offset = buf_slice["offset"]
            for i in range(img_slice["mip_count"]):
                m = i + img_slice["mip_start"]
                for j in range(img_slice["array_count"]):
                    a = j + img_slice["array_start"]
                    s = a + m * img.full_slice["array_count"]
                    buf_footprint = img.cpu_footprints[s]  # image footprint in a continuous buffer
                    width, height, depth = buf_footprint.dim
                    subresource = VkImageSubresourceLayers(VK_IMAGE_ASPECT_COLOR_BIT, m, a, 1)
                    regions.append(VkBufferImageCopy(buf_offset, 0, 0, subresource, (0,0,0), VkExtent3D(width, height, depth)))
                    buf_offset += buf_footprint.size
            return regions
        def get_image_image_regions(src, src_slice, dst, dst_slice):
            assert src_slice["mip_count"] == dst_slice["mip_count"] and src_slice["array_count"] == dst_slice["array_count"], "Can not copy different number of mips or arrays"
            regions = []
            for i in range(src_slice["mip_count"]):
                sm = i + src_slice["mip_start"]
                dm = i + dst_slice["mip_start"]
                for j in range(src_slice["array_count"]):
                    sa = j + src_slice["array_start"]
                    da = j + dst_slice["array_start"]
                    ss = sa + sm * src.full_slice["array_count"]
                    ds = da + dm * dst.full_slice["array_count"]
                    src_footprint = src.footprints[ss]
                    dst_footprint = dst.footprints[ds]
                    assert src_footprint.dim == dst_footprint.dim, "Correspondent subresources copied need to have same extension"
                    width, height, depth = src_footprint.dim
                    s_subresource = VkImageSubresourceLayers(VK_IMAGE_ASPECT_COLOR_BIT, sm, sa, 1)
                    d_subresource = VkImageSubresourceLayers(VK_IMAGE_ASPECT_COLOR_BIT, dm, da, 1)
                    regions.append(VkImageCopy(s_subresource, (0,0,0), d_subresource, (0,0,0), VkExtent3D(width, height, depth)))
            return regions

        if src.is_buffer and dst.is_buffer:
            assert src_slice["size"] == dst_slice["size"], "Can not copy different sizes buffers"
            region = VkBufferCopy(src_slice["offset"], dst_slice["offset"], src_slice["size"])
            vkCmdCopyBuffer(vk_cmdList, src.vk_resource, dst.vk_resource, 1, region)
            return
        if src.is_buffer and not dst.is_buffer:
            regions = get_buffer_image_regions(src, src_slice, dst, dst_slice)
            dst.add_barrier(vk_cmdList, dst_slice, ResourceState(
                vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
                vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                vk_layout=VK_IMAGE_LAYOUT_GENERAL,
                queue_index=VK_QUEUE_FAMILY_IGNORED
            ))
            vkCmdCopyBufferToImage(vk_cmdList, src.vk_resource, dst.vk_resource, VK_IMAGE_LAYOUT_GENERAL, len(regions),
                                   regions)
            return
        if not src.is_buffer and dst.is_buffer:
            regions = get_buffer_image_regions(dst, dst_slice, src, src_slice)
            src.add_barrier(vk_cmdList, src_slice, ResourceState(
                vk_access=VK_ACCESS_TRANSFER_READ_BIT,
                vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                vk_layout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                queue_index=VK_QUEUE_FAMILY_IGNORED
            ))
            vkCmdCopyImageToBuffer(vk_cmdList, src.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.vk_resource, len(regions),
                                   regions)
            return
        if True:  # image to image
            regions = get_image_image_regions(src, src_slice, dst, dst_slice)
            src.add_barrier(vk_cmdList, src_slice, ResourceState(
                vk_access=VK_ACCESS_TRANSFER_READ_BIT,
                vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                vk_layout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                queue_index=VK_QUEUE_FAMILY_IGNORED
            ))
            dst.add_barrier(vk_cmdList, dst_slice, ResourceState(
                vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
                vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                vk_layout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                queue_index=VK_QUEUE_FAMILY_IGNORED
            ))
            vkCmdCopyImage(vk_cmdList, src.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, len(regions), regions)

    @staticmethod
    def copy(device, src, src_slice, dst, dst_slice):
        # torch.cuda.synchronize()  # just in case any pending torch write to gpu shared
        cmdList : CommandBufferWrapper = device.create_cmdList(VK_QUEUE_GRAPHICS_BIT)
        ResourceData.copy_from_to(cmdList.vk_cmdList, src, src_slice, dst, dst_slice)
        cmdList.flush_and_wait()

    # _FORMAT_TYPE_2_NP_TYPE = {
    #     'b': 'u1',
    #     'f': '<f4',
    #     'i': '<i4',
    #     'u': '<u4'
    # }


class ResourceWrapper:
    """
    Wrapper for a vk resource, memory and and initial state.
    """
    def __init__(self,
                 resource_data: ResourceData,
                 resource_slice: Dict[str, int] = None
                 ):
        self.resource_data = resource_data
        self.vk_view = None
        self.vk_device = self.resource_data.vk_device
        if resource_slice is None:
            resource_slice = resource_data.full_slice
        self.current_slice = resource_slice
        self.is_readonly = False

    # @trace_destroying
    def __del__(self):
        # Destroy view if any
        if self.vk_view is not None:
            if self.resource_data.is_buffer:
                vkDestroyBufferView(self.vk_device, self.vk_view, None)
            else:
                vkDestroyImageView(self.vk_device, self.vk_view, None)
        self.resource_data = None
        self.vk_device = None
        self.vk_view = None
        self.gpu_version = None
        self.cpu_version = None

    def get_device_ptr(self):
        return self.resource_data.get_device_ptr(self.current_slice)

    def get_cuda_ptr(self):
        return self.resource_data.get_cuda_ptr(self.current_slice)

    def get_host_ptr(self):
        return self.resource_data.get_host_ptr(self.current_slice)

    def get_tensor(self):
        return self.resource_data.get_tensor(self.current_slice)

    def as_readonly(self):
        new_slice = dict(self.current_slice)
        rw = ResourceWrapper(self.resource_data, new_slice)
        rw.is_readonly = True
        return rw

    def _clone_with_slice(self, new_slice):
        rw = ResourceWrapper(self.resource_data, new_slice)
        return rw

    def slice_mips(self, mip_start, mip_count):
        new_slice = dict(self.current_slice)
        new_slice["mip_start"] = self.current_slice["mip_start"] + mip_start
        new_slice["mip_count"] = mip_count
        return self._clone_with_slice(new_slice)

    def slice_array(self, array_start, array_count):
        new_slice = dict(self.current_slice)
        new_slice["array_start"] = self.current_slice["array_start"] + array_start
        new_slice["array_count"] = array_count
        return self._clone_with_slice(new_slice)

    def subresource(self, mip, layer):
        new_slice = dict(self.current_slice)
        new_slice["mip_start"] = self.current_slice["mip_start"] + mip
        new_slice["mip_count"] = 1
        new_slice["array_start"] = self.current_slice["array_start"] + layer
        new_slice["array_count"] = 1
        return self._clone_with_slice(new_slice)

    def slice_buffer(self, offset, size):
        new_slice = dict(self.current_slice)
        new_slice["offset"] = self.current_slice["offset"] + offset
        new_slice["size"] = size
        return self._clone_with_slice(new_slice)

    def is_single(self):
        return self.resource_data.is_buffer or (self.current_slice['mip_count'] == 1 and self.current_slice['array_count'] == 1)

    # def is_homogeneous(self):
    #     return self.resource_data.is_buffer or (self.current_slice['mip_count'] == 1)

    def _load_from_resource(self, w_device, resource):
        resource: ResourceWrapper
        ResourceData.copy(w_device, resource.resource_data, resource.current_slice, self.resource_data, self.current_slice)
        return self

    def _load_from_cuda_ptr(self, src_ptr):
        assert self.resource_data.is_buffer, "Can not transfer directly to a non-buffer resource. Consider using staging buffers"
        # if self.resource_data.is_buffer:
        dst_ptr = self.get_cuda_ptr()
        size = self.current_slice["size"]
        cuda.cuMemcpy(dst_ptr, src_ptr, size)
        # else:
        #     dst_ptr = self.resource_data.get_cuda_ptr()
        #     mip_start = self.current_slice["mip_start"]
        #     mip_count = self.current_slice["mip_count"]
        #     array_start = self.current_slice["array_start"]
        #     array_count = self.current_slice["array_count"]
        #     full_array_count = self.resource_data.full_slice["array_count"]
        #     data_offset = 0
        #     for mip in range(mip_start, mip_start + mip_count):
        #         for arr in range(array_start, array_start + array_count):
        #             subresource_index = arr + mip * full_array_count
        #             footprint, offset = self.resource_data.get_footprint_and_offset(subresource_index)
        #             cpu_footprint = self.resource_data.cpu_footprints[subresource_index]
        #             if cpu_footprint.size == footprint.size:
        #                 cuda.cuMemcpy(dst_ptr + offset, src_ptr + data_offset, cpu_footprint.size)
        #                 data_offset += cpu_footprint.size
        #             else:
        #                 for z in range(cpu_footprint.dim[2]):
        #                     for y in range(cpu_footprint.dim[1]):
        #                         cuda.cuMemcpy(dst_ptr + offset, src_ptr + data_offset, cpu_footprint.row_pitch)
        #                         data_offset += cpu_footprint.row_pitch
        #                         offset += footprint.row_pitch
        #                     offset += footprint.slice_pitch - footprint.row_pitch * cpu_footprint.dim[1]
        return self

    def load(self, w_device, src):
        if src is self:
            return self
        if isinstance(src, ResourceWrapper):
            return self._load_from_resource(w_device, src)
        if isinstance(src, list):
            src = np.array(src, dtype=np.float32())
            src.setflags(write=True)
        if hasattr(src, "__cuda_array_interface__"):
            ptr = src.__cuda_array_interface__['data'][0]
            return self._load_from_cuda_ptr(ptr)
        if hasattr(src, "__array_interface__"):
            ptr = src.__array_interface__['data'][0]
            return self._load_from_cuda_ptr(ptr)
        try:
            a = np.frombuffer(src, dtype=np.uint8)
            # a.setflags(write=True)
            ptr = a.__array_interface__['data'][0]
            return self._load_from_cuda_ptr(ptr)
        except:
            pass
        try:
            t = torch.frombuffer(src, dtype=torch.uint8)  # try to wrap data with a tensor
            ptr = t.data_ptr()
            return self._load_from_cuda_ptr(ptr)
        except:
            pass
        try:
            t = torch.from_dlpack(src)
            ptr = t.data_ptr()
            return self._load_from_cuda_ptr(ptr)
        except:
            # return self._load_from_bytes(src.numpy().data.cast('b'))
            raise Exception('No supported loading from '+str(type(src)))

    def _save_to_resource(self, w_device, resource):
        resource: ResourceWrapper
        ResourceData.copy(w_device, self.resource_data, self.current_slice, resource.resource_data, resource.current_slice)
        return self

    def _save_to_cuda_ptr(self, dst_ptr):
        assert self.resource_data.is_buffer, "Can not transfer directly from a non-buffer resource. Consider using staging buffers"
        # if self.resource_data.is_buffer:
        src_ptr = self.get_cuda_ptr()
        size = self.current_slice["size"]
        cuda.cuMemcpy(dst_ptr, src_ptr, size)
        # else:
        #     src_ptr = self.resource_data.get_cuda_ptr()  # Full slice cuda ptr
        #     mip_start = self.current_slice["mip_start"]
        #     mip_count = self.current_slice["mip_count"]
        #     array_start = self.current_slice["array_start"]
        #     array_count = self.current_slice["array_count"]
        #     full_array_count = self.resource_data.full_slice["array_count"]
        #     data_offset = 0
        #     for mip in range(mip_start, mip_start + mip_count):
        #         for arr in range(array_start, array_start + array_count):
        #             subresource_index = arr + mip * full_array_count
        #             footprint, offset = self.resource_data.get_footprint_and_offset(subresource_index)
        #             cpu_footprint = self.resource_data.cpu_footprints[subresource_index]
        #             if cpu_footprint.size == footprint.size:
        #                 cuda.cuMemcpy(dst_ptr + data_offset, src_ptr + offset, cpu_footprint.size)
        #                 data_offset += cpu_footprint.size
        #             else:
        #                 for z in range(cpu_footprint.dim[2]):
        #                     for y in range(cpu_footprint.dim[1]):
        #                         cuda.cuMemcpy(dst_ptr + data_offset, src_ptr + offset, cpu_footprint.row_pitch)
        #                         data_offset += cpu_footprint.row_pitch
        #                         offset += footprint.row_pitch
        #                     offset += footprint.slice_pitch - cpu_footprint.dim[1] * footprint.row_pitch
        return self

    def save(self, w_device, dst):
        if dst is self:
            return self
        if isinstance(dst, ResourceWrapper):
            return self._save_to_resource(w_device, dst)
        if hasattr(dst, "__cuda_array_interface__"):
            ptr = dst.__cuda_array_interface__['data'][0]
            return self._save_to_cuda_ptr(ptr)
        if hasattr(dst, "__array_interface__"):
            ptr = dst.__array_interface__['data'][0]
            return self._save_to_cuda_ptr(ptr)
        try:
            t = torch.frombuffer(dst, dtype=torch.uint8)  # try to wrap data with a tensor
            ptr = t.data_ptr()
            return self._save_to_cuda_ptr(ptr)
        except:
            pass
        try:
            t = torch.from_dlpack(dst)  # try to wrap data with a tensor
            ptr = t.data_ptr()
            return self._save_to_cuda_ptr(ptr)
        except:
            raise Exception('No supported save to type '+str(type(dst)))

    def add_barrier(self, vk_cmdList, state: ResourceState):
        self.resource_data.add_barrier(vk_cmdList, self.current_slice, state)

    def get_view(self):
        if self.vk_view is None:
            if self.resource_data.is_buffer:
                self.vk_view = vkCreateBufferView(self.resource_data.vk_device, VkBufferViewCreateInfo(
                    buffer=self.resource_data.vk_resource,
                    offset=self.current_slice["offset"],
                    range=self.current_slice["size"]
                ), None)
            else:
                self.vk_view = vkCreateImageView(self.resource_data.vk_device, VkImageViewCreateInfo(
                    image=self.resource_data.vk_resource,
                    viewType=self.resource_data.vk_description.imageType,
                    flags=0,
                    format=self.resource_data.vk_description.format,
                    components=VkComponentMapping(
                        r=VK_COMPONENT_SWIZZLE_IDENTITY,
                        g=VK_COMPONENT_SWIZZLE_IDENTITY,
                        b=VK_COMPONENT_SWIZZLE_IDENTITY,
                        a=VK_COMPONENT_SWIZZLE_IDENTITY,
                    ),
                    subresourceRange=ResourceData.get_subresources(self.current_slice)
                ), None)
        return self.vk_view

    def get_size(self):
        """
        Gets the size in bytes of the memory occupied by this resource.
        Is only valid if the slice is contiguous in memory.
        """
        return self.resource_data.cpu_size

    def get_storage_size(self):
        return self.resource_data.w_memory.size

    def get_footprints(self):
        return (
            self.resource_data.get_cpu_footprint_and_offset(i)[0]
            for i in self.resource_data.get_subresource_indices(self.current_slice)
        )


class SamplerWrapper:
    def __init__(self, w_device, vk_sampler):
        self.w_device = w_device
        self.vk_sampler = vk_sampler

    def __del__(self):
        vkDestroySampler(self.w_device.vk_device, self.vk_sampler, None)


class ShaderHandlerWrapper:
    def __init__(self):
        self.handle = None

    def get_handle(self):
        return self.handle

    def _set_handle(self, handle):
        self.handle = handle


class PipelineBindingWrapper:
    def __init__(self, w_device, pipeline_type):
        self.w_device = w_device
        self.pipeline_type = pipeline_type
        self.descriptor_sets_description = [[], [], [], []]
        self.active_set = 0
        self.active_descriptor_set = self.descriptor_sets_description[0]  # by default activate global set
        self.shaders = {}
        self.push_constants = { }
        self.rt_shaders = []  # shader stages
        self.rt_groups = []  # groups for hits
        self.rt_group_handles = []  # handles computed for each group
        self.max_recursion_depth = 1
        self.initialized = False
        self.descriptor_set_layouts = []
        self.descriptor_sets = None
        self.descriptor_pool = None
        self.pipeline_object = None
        self.pipeline_layout = None
        self.current_cmdList = None
        if self.pipeline_type == VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO:
            self.point = VK_PIPELINE_BIND_POINT_COMPUTE
            self.__valid_stages = [VK_SHADER_STAGE_COMPUTE_BIT]
        elif self.pipeline_type == VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO:
            self.point = VK_PIPELINE_BIND_POINT_GRAPHICS
            self.__valid_stages = [
                VK_SHADER_STAGE_VERTEX_BIT,
                VK_SHADER_STAGE_FRAGMENT_BIT,
                VK_SHADER_STAGE_GEOMETRY_BIT,
                VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
            ]
        else:
            self.point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR
            self.__valid_stages = [
                VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                VK_SHADER_STAGE_MISS_BIT_KHR,
                VK_SHADER_STAGE_CALLABLE_BIT_KHR
            ]
        self.active_stage = self.__valid_stages[0]

    def _vk_destroy(self):
        # Destroy desciptor sets
        if self.w_device is None:
            return
        vkDestroyDescriptorPool(self.w_device.vk_device, self.descriptor_pool, None)
        # Destroy layouts
        [vkDestroyDescriptorSetLayout(self.w_device.vk_device, dl, None) for dl in self.descriptor_set_layouts]
        # Destroy pipeline layout
        if self.pipeline_layout:
            vkDestroyPipelineLayout(self.w_device.vk_device, self.pipeline_layout, None)
        # Destroy pipeline object
        if self.pipeline_object:
            vkDestroyPipeline(self.w_device.vk_device, self.pipeline_object, None)
        self.descriptor_sets_description = [[], [], [], []]
        self.active_set = 0
        self.active_descriptor_set = None
        self.shaders = {}
        self.descriptor_set_layouts = []
        self.descriptor_sets = None
        self.descriptor_pool = None
        self.pipeline_object = None
        self.pipeline_layout = None
        self.current_cmdList = None
        self.w_device = None

    @trace_destroying
    def __del__(self):
        self._vk_destroy()

    def descriptor_set(self, set_slot):
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        self.active_set = set_slot
        self.active_descriptor_set = self.descriptor_sets_description[set_slot]

    def shader_stage(self, shader_stage: int):
        assert shader_stage in self.__valid_stages, "Shader stage not valid for this pipeline"
        self.active_stage = shader_stage

    def binding(self, slot, vk_descriptor_type, count, resolver):
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        self.active_descriptor_set.append(
            (slot, self.active_stage, vk_descriptor_type, count, resolver)
        )

    def add_constant_range(self, offset, size, layout):
        assert self.active_stage not in self.push_constants, "Can not have more than one range of constants per stage"
        self.push_constants[self.active_stage] = (offset, size, layout, bytearray(size))

    def load_shader(self, w_shader) -> int:
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        w_shader.vk_stage_info.stage = self.active_stage
        if self.pipeline_type == VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR:
            self.rt_shaders.append(w_shader)
            return len(self.rt_shaders)-1
        else:
            self.shaders[self.active_stage] = w_shader
            return len(self.shaders) - 1

    def create_hit_group(self,
                         closest_hit_index: int = None,
                         any_hit_index: int = None,
                         intersection_index: int = None):
        type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR \
            if intersection_index is None \
            else VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
        self.rt_groups.append(VkRayTracingShaderGroupCreateInfoKHR(
            type=type,
            generalShader=VK_SHADER_UNUSED_KHR,
            closestHitShader=VK_SHADER_UNUSED_KHR if closest_hit_index is None else closest_hit_index,
            anyHitShader=VK_SHADER_UNUSED_KHR if any_hit_index is None else any_hit_index,
            intersectionShader=VK_SHADER_UNUSED_KHR if intersection_index is None else intersection_index
        ))
        s = ShaderHandlerWrapper()
        self.rt_group_handles.append(s)
        return s

    def create_general_group(self, shader_index: int):
        type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR
        self.rt_groups.append(VkRayTracingShaderGroupCreateInfoKHR(
            type=type,
            generalShader=shader_index,
            closestHitShader=VK_SHADER_UNUSED_KHR,
            anyHitShader=VK_SHADER_UNUSED_KHR,
            intersectionShader=VK_SHADER_UNUSED_KHR
        ))
        s = ShaderHandlerWrapper()
        self.rt_group_handles.append(s)
        return s

    def set_max_recursion(self, depth):
        self.max_recursion_depth = depth

    def _build_objects(self):
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        # Builds the descriptor sets layouts
        descriptor_set_layout_bindings = [[], [], [], []]
        descriptor_set_layout_bindings_bound = [[],[],[],[]]
        counting_by_type = {
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: 1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: 1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: 1,
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: 1,
        }
        var_count_info = []
        for level in range(4):
            has_variable_descriptor = False
            for slot, vk_stage, vk_descriptor_type, count, resolver in self.descriptor_sets_description[level]:
                effect_count = 100 if count == -1 else count
                lb = VkDescriptorSetLayoutBinding(
                    slot,
                    vk_descriptor_type,
                    effect_count, # for unbound descriptor set array
                    vk_stage
                )
                # bound = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT \
                # bound = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT\
                #     if count == -1 else \
                #     VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                bound = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT  \
                    if count == -1 else \
                    0
                if count == -1:
                    has_variable_descriptor = True
                descriptor_set_layout_bindings[level].append(lb)
                descriptor_set_layout_bindings_bound[level].append(bound)
                counting_by_type[vk_descriptor_type] += effect_count * self.w_device.get_number_of_frames()
            var_count_info.append(0 if not has_variable_descriptor else 100)
        self.descriptor_pool = vkCreateDescriptorPool(self.w_device.vk_device, VkDescriptorPoolCreateInfo(
            maxSets=4 * max(1, self.w_device.get_number_of_frames()),
            poolSizeCount=6,
            pPoolSizes=[VkDescriptorPoolSize(t, c) for t, c in counting_by_type.items()],
            flags=VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
        ), pAllocator=None)

        self.descriptor_set_layouts = []
        for level, lb in enumerate(descriptor_set_layout_bindings):
            bound_info = VkDescriptorSetLayoutBindingFlagsCreateInfo(
                pBindingFlags=descriptor_set_layout_bindings_bound[level],
                bindingCount=len(descriptor_set_layout_bindings_bound[level])
            )
            dslci = VkDescriptorSetLayoutCreateInfo(
                pNext = bound_info,
                bindingCount=len(lb),
                pBindings=lb,
                flags=VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
            )
            self.descriptor_set_layouts.append(
                vkCreateDescriptorSetLayout(self.w_device.vk_device,
                                        dslci, None))

        # Builds the descriptor sets (one group for each frame)
        descriptor_set_layouts_per_frame = self.descriptor_set_layouts * self.w_device.get_number_of_frames()
        var_count_info = var_count_info * self.w_device.get_number_of_frames()
        var_count = VkDescriptorSetVariableDescriptorCountAllocateInfo(
            descriptorSetCount=len(var_count_info),
            pDescriptorCounts=var_count_info,
        )
        allocate_info = VkDescriptorSetAllocateInfo(
            pNext = None, # var_count,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=len(descriptor_set_layouts_per_frame),
            pSetLayouts=descriptor_set_layouts_per_frame)
        self.descriptor_sets = vkAllocateDescriptorSets(self.w_device.vk_device, allocate_info)
        # Builds pipeline object
        push_constant_ranges = [VkPushConstantRange(
            size=size,
            offset=offset,
            stageFlags=stage
        ) for stage, (offset, size, layout, data) in self.push_constants.items()]
        pipeline_layout_info = VkPipelineLayoutCreateInfo(
            setLayoutCount=4,
            pSetLayouts=self.descriptor_set_layouts,
            pushConstantRangeCount=len(push_constant_ranges),
            pPushConstantRanges=push_constant_ranges
        )
        self.pipeline_layout = vkCreatePipelineLayout(self.w_device.vk_device, pipeline_layout_info, None)
        if self.pipeline_type == VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO:
            assert VK_SHADER_STAGE_COMPUTE_BIT in self.shaders, "Error, no compute shader bound!"
            self.pipeline_object = vkCreateComputePipelines(self.w_device.vk_device, VK_NULL_HANDLE, 1,
                                                            [
                                                                VkComputePipelineCreateInfo(
                                                                    layout=self.pipeline_layout,
                                                                    stage=self.shaders[
                                                                        VK_SHADER_STAGE_COMPUTE_BIT
                                                                    ].vk_stage_info
                                                                )], None)[0]
        elif self.pipeline_type == VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO:
            pass
        elif self.pipeline_type == VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR:
            self.pipeline_object = self.w_device.vkCreateRayTracingPipelines(
                self.w_device.vk_device, VK_NULL_HANDLE,
                VK_NULL_HANDLE, 1, VkRayTracingPipelineCreateInfoKHR(
                                        layout=self.pipeline_layout,
                                        stageCount=len(self.rt_shaders),
                                        pStages=[s.vk_stage_info for s in self.rt_shaders],
                                        groupCount=len(self.rt_groups),
                                        pGroups=self.rt_groups,
                                        maxPipelineRayRecursionDepth=self.max_recursion_depth
                                    ), None)[0]
            shader_handle_size = self.w_device.raytracing_properties.shaderGroupHandleSize
            all_handles = bytearray(shader_handle_size * len(self.rt_groups))
            self.w_device.vkGetRayTracingShaderGroupHandles(
                self.w_device.vk_device,
                self.pipeline_object,
                0,
                len(self.rt_groups),
                len(self.rt_groups)*shader_handle_size,
                ffi.from_buffer(all_handles)
            )
            for i, s in enumerate(self.rt_group_handles):
                s._set_handle(all_handles[i*shader_handle_size:(i+1)*shader_handle_size])
        self.initialized = True

    def _set_at_cmdList(self, vk_cmdList, queue_index):
        frame = self.w_device.get_frame_index()
        self.current_cmdList = vk_cmdList
        self.current_queue_index = queue_index
        vkCmdBindPipeline(vk_cmdList, self.point, self.pipeline_object)

    def _solve_resolver_as_buffers(self, buffer: ResourceWrapper, vk_descriptor_type):
        if buffer is None:
            # NULL DESCRIPTOR
            return VkDescriptorBufferInfo(
                buffer=None,
                offset=0,
                range=UINT64_MAX
            )
        if vk_descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            return None
        else:
            return VkDescriptorBufferInfo(
                buffer=buffer.resource_data.vk_resource,
                offset=buffer.current_slice["offset"],
                range=buffer.current_slice["size"]
            )

    __SHADER_STAGE_2_PIPELINE_STAGE = {
        VK_SHADER_STAGE_VERTEX_BIT: VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT: VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT: VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
        VK_SHADER_STAGE_MESH_BIT_NV: VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
        VK_SHADER_STAGE_COMPUTE_BIT: VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_MISS_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    }

    def _solve_resolver_as_image(self, t, vk_descriptor_type, vk_shader_stage):

        vk_stage = PipelineBindingWrapper.__SHADER_STAGE_2_PIPELINE_STAGE[vk_shader_stage]

        if vk_descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            image, sampler = t
        else:
            image = t
            sampler = None

        if vk_descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE or \
           vk_descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            vk_access = VK_ACCESS_SHADER_READ_BIT
            vk_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        else:
            if image.w_resource.is_readonly:
                vk_access = VK_ACCESS_SHADER_READ_BIT
            else:
                vk_access = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
            vk_layout = VK_IMAGE_LAYOUT_GENERAL

        image.w_resource.add_barrier(self.current_cmdList, ResourceState(
            vk_access=vk_access,
            vk_layout=vk_layout,
            vk_stage=vk_stage,
            queue_index=self.current_queue_index
        ))
        return VkDescriptorImageInfo(
            imageView=image.w_resource.get_view(),
            imageLayout=vk_layout,
            sampler=None if sampler is None else sampler.vk_sampler
        )

    def _update_level(self, level):
        frame = self.w_device.get_frame_index()
        descriptorWrites = []
        for slot, vk_stage, vk_descriptor_type, count, resolver in self.descriptor_sets_description[level]:
            is_buffer = vk_descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER or \
                        vk_descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or\
                        vk_descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
            next = None
            if vk_descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:  # set next pointer
                next = VkWriteDescriptorSetAccelerationStructureKHR(
                    accelerationStructureCount=1,
                    pAccelerationStructures=[resolver()[0].w_resource.resource_data.ads]
                )
            if is_buffer:
                descriptors = [self._solve_resolver_as_buffers(b.w_resource if b is not None else None, vk_descriptor_type) for b in resolver()]
            else:
                descriptors = [self._solve_resolver_as_image(t, vk_descriptor_type, vk_stage) for t in resolver()]
            dw = VkWriteDescriptorSet(
                pNext=next,
                dstSet=self.descriptor_sets[frame*4 + level],
                dstBinding=slot,
                descriptorType=vk_descriptor_type,
                descriptorCount=len(descriptors),
                pBufferInfo= descriptors if is_buffer and not next else None,
                pImageInfo= descriptors if not is_buffer else None
            )
            descriptorWrites.append(dw)
        vkUpdateDescriptorSets(
            device=self.w_device.vk_device,
            descriptorWriteCount=len(descriptorWrites),
            pDescriptorWrites=descriptorWrites,
            descriptorCopyCount=0,
            pDescriptorCopies=None
        )
        vkCmdBindDescriptorSets(
            commandBuffer=self.current_cmdList,
            pipelineBindPoint=self.point,
            layout=self.pipeline_layout,
            firstSet=level,
            descriptorSetCount=1,
            pDescriptorSets=[self.descriptor_sets[frame * 4 + level]],
            dynamicOffsetCount=0,
            pDynamicOffsets=None
        )


class CommandListState(Enum):
    NONE = 0
    INITIAL = 1
    RECORDING = 2
    EXECUTABLE = 3
    SUBMITTED = 4
    FINISHED = 5


class CommandBufferWrapper:
    """
    Wrapper for a command list.
    """

    def __init__(self, vk_cmdList, pool):
        self.vk_cmdList = vk_cmdList
        self.pool : CommandPoolWrapper = pool
        self.__is_frozen = False
        self.state = CommandListState.INITIAL
        self.current_pipeline = None
        self.shader_groups_size = self.pool.device.raytracing_properties.shaderGroupHandleSize

    @trace_destroying
    def __del__(self):
        self.current_pipeline = None
        self.current_program = None
        self.pool = None

    vkCmdBuildAccelerationStructures=None
    vkBuildAccelerationStructures = None
    vkCmdTraceRays=None

    def begin(self):
        assert self.state == CommandListState.INITIAL, "Error, to begin a cmdList should be in initial state"
        vkBeginCommandBuffer(self.vk_cmdList, VkCommandBufferBeginInfo())
        self.state = CommandListState.RECORDING

    def end(self):
        assert self.state == CommandListState.RECORDING, "Error, to end a cmdList should be in recording"
        vkEndCommandBuffer(self.vk_cmdList)
        self.state = CommandListState.EXECUTABLE
        # self.current_pipeline = None

    def reset(self):
        assert not self.__is_frozen, "Can not reset a frozen cmdList"
        assert self.state == CommandListState.EXECUTABLE, "Error, to reset a cmdList should be in executable state"
        vkResetCommandBuffer(self.vk_cmdList, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)
        self.state = CommandListState.INITIAL

    def flush_and_wait(self):
        self.pool.flush([self])

    def freeze(self):
        if self.is_frozen():
            return
        self.end()
        self.__is_frozen = True

    def close(self):
        self.end()
        self.state = CommandListState.FINISHED

    def is_closed(self):
        return self.state == CommandListState.FINISHED

    def is_frozen(self):
        return self.__is_frozen

    def clear_buffer(self, w_buffer: ResourceWrapper, value: int = 0):
        vkCmdFillBuffer(self.vk_cmdList, w_buffer.resource_data.vk_resource,
                        w_buffer.current_slice["offset"], w_buffer.current_slice["size"],
                        value)

    def clear_color(self, w_image: ResourceWrapper, color):
        if not isinstance(color, list) and not isinstance(color, tuple):
            color = list(color)

        image = w_image.resource_data.vk_resource

        new_access, new_stage, new_layout, new_queue = w_image.resource_data.current_state
        if new_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL or \
                new_layout != VK_IMAGE_LAYOUT_GENERAL or \
                new_layout != VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR:
            new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL

        w_image.add_barrier(self.vk_cmdList, ResourceState(
            vk_access=new_access,
            vk_stage=new_stage,
            vk_layout=new_layout,
            queue_index=new_queue
        ))
        vkCmdClearColorImage(self.vk_cmdList, image, new_layout, VkClearColorValue(color), 1,
                             [ResourceData.get_subresources(w_image.current_slice)])

    def copy(self, src: ResourceWrapper, dst: ResourceWrapper):
        ResourceData.copy_from_to(self.vk_cmdList, src.resource_data, src.current_slice, dst.resource_data, dst.current_slice)

    def blit_image(self, w_src: ResourceWrapper, w_dst: ResourceWrapper, filter: int):
        w_src.add_barrier(self.vk_cmdList, ResourceState(
            vk_access=VK_ACCESS_TRANSFER_READ_BIT,
            vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk_layout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            queue_index=VK_QUEUE_FAMILY_IGNORED))
        w_dst.add_barrier(self.vk_cmdList, ResourceState(
            vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
            vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk_layout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            queue_index=VK_QUEUE_FAMILY_IGNORED))

        src_index = w_src.current_slice["array_start"] + w_src.current_slice["mip_start"] * w_src.resource_data.full_slice["array_count"]
        src_footprint, _ = w_src.resource_data.get_cpu_footprint_and_offset(src_index)
        dst_index = w_dst.current_slice["array_start"] + w_dst.current_slice["mip_start"] * w_dst.resource_data.full_slice["array_count"]
        dst_footprint, _ = w_dst.resource_data.get_cpu_footprint_and_offset(dst_index)

        src_w, src_h, src_d = src_footprint.dim
        dst_w, dst_h, dst_d = dst_footprint.dim

        src_regions = [VkOffset3D(0,0,0), VkOffset3D(src_w, src_h, src_d)]
        dst_regions = [VkOffset3D(0,0,0), VkOffset3D(dst_w, dst_h, dst_d)]

        vkCmdBlitImage(self.vk_cmdList, w_src.resource_data.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, w_dst.resource_data.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                       [
                           VkImageBlit(
                               ResourceData.get_subresource_layers(w_src.current_slice),
                               src_regions,
                               ResourceData.get_subresource_layers(w_dst.current_slice),
                               dst_regions
                           )
                       ], filter)

    # def copy_buffer_to_image(self, w_src: ResourceWrapper, w_dst: ResourceWrapper):
    #     w_dst.add_barrier(self.vk_cmdList, ResourceState(
    #         vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
    #         vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    #         vk_layout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    #         queue_index=VK_QUEUE_FAMILY_IGNORED))
    #     subresource_index = w_dst.current_slice["array_start"] + w_dst.current_slice["mip_start"] * \
    #                         w_dst.resource_data.full_slice["array_count"]
    #     footprint, offset = w_dst.resource_data.get_staging_footprint_and_offset(subresource_index)
    #     w, h, d = footprint.dim
    #     vkCmdCopyBufferToImage(self.vk_cmdList, w_src.resource_data.vk_resource, w_dst.resource_data.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,[
    #         VkBufferImageCopy(w_src.current_slice["offset"], 0, 0, ResourceWrapper.get_subresource_layers(w_dst.current_slice), VkOffset3D(0,0,0), VkExtent3D(w, h, d))
    #     ])
    #
    # def copy_image(self, w_src: ResourceWrapper, w_dst: ResourceWrapper):
    #     w_src.add_barrier(self.vk_cmdList, ResourceState(
    #         vk_access=VK_ACCESS_TRANSFER_READ_BIT,
    #         vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    #         vk_layout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    #         queue_index=VK_QUEUE_FAMILY_IGNORED))
    #     w_dst.add_barrier(self.vk_cmdList, ResourceState(
    #         vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
    #         vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    #         vk_layout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    #         queue_index=VK_QUEUE_FAMILY_IGNORED))
    #     subresource_index = w_src.current_slice["array_start"] + w_src.current_slice["mip_start"] * w_src.resource_data.full_slice["array_count"]
    #     footprint, offset = w_src.resource_data.get_staging_footprint_and_offset(subresource_index)
    #     w, h, d = footprint.dim
    #     vkCmdCopyImage(self.vk_cmdList, w_src.resource_data.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    #                    w_dst.resource_data.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, [
    #             VkImageCopy(
    #                 ResourceWrapper.get_subresource_layers(w_src.current_slice),
    #                 VkOffset3D(0,0,0),
    #                 ResourceWrapper.get_subresource_layers(w_dst.current_slice),
    #                 VkOffset3D(0,0,0), VkExtent3D(w, h, d)
    #             )
    #                    ])

    def set_pipeline(self, pipeline: PipelineBindingWrapper):
        self.current_pipeline = pipeline
        pipeline._set_at_cmdList(self.vk_cmdList, self.pool.queue_index)

    def update_constants(self, stage, **fields):
        assert stage in self.current_pipeline.push_constants, "Current pipeline have no constant for the specified stage"
        offset, size, layout, data = self.current_pipeline.push_constants[stage]
        for f, v in fields.items():
            assert f in layout, f"Field {f} was not found in that stage constants"
            field_offset, field_size, field_type = layout[f]  # layout
            data[field_offset:field_offset + field_size] = v
        vkCmdPushConstants(
            self.vk_cmdList,
            self.current_pipeline.pipeline_layout,
            stage, offset, size, ffi.from_buffer(data)
        )

    def update_bindings_level(self, level):
        self.current_pipeline._update_level(level)

    def dispatch_groups(self, dimx, dimy, dimz):
        # vkCmdPipelineBarrier(self.vk_cmdList,
        #                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        #                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        #                      0, 1, VkMemoryBarrier(
        #         srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        #         dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
        #     ), 0, 0, 0, 0)
        vkCmdDispatch(self.vk_cmdList, dimx, dimy, dimz)
        # vkCmdPipelineBarrier(self.vk_cmdList,
        #                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        #                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        #                      0, 1, VkMemoryBarrier(
        #         srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        #         dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
        #     ), 0, 0, 0, 0)

    def _get_strided_device_address(self, w_resource: ResourceWrapper, stride):
        if w_resource is None:
            return w_resource
        address = self.pool.device._get_device_address(w_resource)
        return VkStridedDeviceAddressRegionKHR(
            address.deviceAddress, stride, w_resource.current_slice["size"]
        )

    def dispatch_rays(self,
                      w_raygen_table: ResourceWrapper,
                      w_raymiss_table: ResourceWrapper,
                      w_rayhit_table: ResourceWrapper,
                      dimx: int, dimy: int, dimz: int):
        CommandBufferWrapper.vkCmdTraceRays(
            self.vk_cmdList,
            self._get_strided_device_address(w_raygen_table, self.shader_groups_size),
            self._get_strided_device_address(w_raymiss_table, self.shader_groups_size),
            self._get_strided_device_address(w_rayhit_table, self.shader_groups_size),
            VkStridedDeviceAddressRegionKHR(),
            dimx, dimy, dimz
        )
        vkCmdPipelineBarrier(self.vk_cmdList, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, VkMemoryBarrier(
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT
            ), 0, 0, 0, 0)

    def build_ads(self,
                  w_ads: ResourceWrapper,
                  ads_info: VkAccelerationStructureBuildGeometryInfoKHR,
                  ads_ranges,
                  w_scratch: ResourceWrapper):
        vkCmdPipelineBarrier(self.vk_cmdList,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1,
                             VkMemoryBarrier(
                                 srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                                 dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                             ), 0, 0, 0, 0)

        build_info = VkAccelerationStructureBuildGeometryInfoKHR(
            type = ads_info.type,
            geometryCount=ads_info.geometryCount,
            pGeometries=ads_info.pGeometries,
            scratchData=self.pool.device._get_device_address(w_scratch),
            dstAccelerationStructure=w_ads.resource_data.ads,
            mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        )
        CommandBufferWrapper.vkCmdBuildAccelerationStructures(
            self.vk_cmdList,
            # self.pool.vk_device,
            # None,
            1,
            build_info,
            [ads_ranges]
        )
        # vkCmdPipelineBarrier(self.vk_cmdList,
        #                      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        #                      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        #                      0, 1,
        #                          VkMemoryBarrier(
        #                         srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        #                         dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        #                         ), 0, 0, 0, 0)


class ShaderStageWrapper:

    def __init__(self, main_function, module, specialization):
        map = []
        offset = 0
        for i, value in enumerate(specialization):
            assert isinstance(value, int), "Can not specialize with other than ints... at least for now..."
            map.append(VkSpecializationMapEntry(constantID=i, offset=offset, size=4))
            offset += 4
        data = ffi.from_buffer(struct.pack('i'*len(map), *specialization))
        vk_specialization = VkSpecializationInfo(pMapEntries=map, pData=data)
        self.vk_stage_info = VkPipelineShaderStageCreateInfo(
            stage=None,  # Attached later
            module=module,
            pName=main_function,
            pSpecializationInfo=vk_specialization
        )

    @staticmethod
    def from_file(device, path, main_function, specialization):
        vk_device = device.vk_device
        if (path, main_function) not in device.loaded_modules:
            with open(path, mode='rb') as f:
                bytecode = f.read(-1)
                info = VkShaderModuleCreateInfo(
                    codeSize=len(bytecode),
                    pCode=bytecode,
                )
                device.loaded_modules[(path, main_function)] = vkCreateShaderModule(
                    device=vk_device,
                    pCreateInfo=info,
                    pAllocator=None
                )
        return ShaderStageWrapper(
            main_function,
            device.loaded_modules[(path, main_function)], specialization)


class CommandPoolWrapper:
    def __init__(self, device, vk_queue, vk_pool, queue_index):
        self.device = device
        self.vk_device = device.vk_device
        self.vk_pool = vk_pool
        self.vk_queue = vk_queue
        self.attached = []  # attached CommandBufferWrapper can be automatically flushed
        self.reusable = []  # commandlists have been submitted and finished that can be reused
        self.queue_index = queue_index

    def _vk_destroy(self):
        if self.device is None:
            return
        if len(self.reusable) > 0:
            vkFreeCommandBuffers(self.vk_device, self.vk_pool, len(self.reusable), self.reusable)
        if self.vk_pool:
            vkDestroyCommandPool(self.vk_device, self.vk_pool, None)
        self.attached = []  # attached CommandBufferWrapper can be automatically flushed
        self.reusable = []  # commandlists have been submitted and finished that can be reused
        self.device = None

    @trace_destroying
    def __del__(self):
        self._vk_destroy()

    def get_cmdList(self):
        """"
        Gets a new command buffer wrapper
        """
        if len(self.reusable) != 0:
            cmdList = self.reusable.pop()
        else:
            # allocate a new one
            cmdList = vkAllocateCommandBuffers(self.vk_device, VkCommandBufferAllocateInfo(
                commandPool=self.vk_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1))[0]
        cmdList_wrapper = CommandBufferWrapper(cmdList, self)
        self.attached.append(cmdList_wrapper)
        cmdList_wrapper.begin()
        return cmdList_wrapper

    def freeze(self, manager):
        manager.freeze()
        self.attached.remove(manager)

    def flush(self, managers=None):
        if managers is None:  # flush all attached (pending) buffers
            managers = self.attached
            self.attached = []
        else:
            for m in managers:
                if not m.is_frozen():
                    self.attached.remove(m)  # remove from attached

        if len(managers) == 0:
            vkQueueWaitIdle(self.vk_queue)
            return  # finished task

        cmdLists = []
        for m in managers:
            if m.state == CommandListState.SUBMITTED:
                raise Exception(
                    "Error! submitting a frozen command list already on gpu. Use wait for sync."
                )
            if m.state == CommandListState.FINISHED:
                raise Exception(
                    "Error! submitting a command list already executed. Use a frozen cmdList to repeat submissions."
                )
            if m.state == CommandListState.RECORDING:
                m.end()
            assert m.state == CommandListState.EXECUTABLE, "Error in command list state"
            cmdLists.append(m.vk_cmdList)
            m.state = CommandListState.SUBMITTED

        if len(cmdLists) > 0:
            vkQueueSubmit(self.vk_queue, 1,

                              VkSubmitInfo(
                                  commandBufferCount=len(cmdLists),
                                  pCommandBuffers=cmdLists
                              )
                          , None)

        vkQueueWaitIdle(self.vk_queue)
        for c in managers:
            if c.is_frozen():
                c.state = CommandListState.EXECUTABLE
            else:
                vkResetCommandBuffer(c.vk_cmdList, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)
                self.reusable.append(c.vk_cmdList)
                c.state = CommandListState.FINISHED


class Event(Enum):
    NONE = 0
    CLOSED = 1


class WindowWrapper:
    def poll_events(self) -> (Event, object):
        pass


class DeviceWrapper:
    def __init__(self, width, height, format, mode, render_usage, enable_validation_layers):
        self.width = width
        self.height = height
        self.__render_target_format = format
        self.mode = mode
        self.enable_validation_layers = enable_validation_layers
        self.render_target_usage = render_usage
        self.__resources = weakref.WeakSet()
        self.__pipelines = weakref.WeakSet()
        self.vk_device = None
        self.__window = None
        self.__instance = None
        self.__callback = None
        self.__surface = None
        self.__physical_device = None
        self.__swapchain = None
        self.__queues = []  # list of queue objects for each queue family (only one queue used for family)
        self.__queue_index = {}  # map from QueueType to queue index covering the functionalities
        self.__main_manager = None  # Main rendezvous queue used for presenting
        self.__main_queue_index = -1
        self.__managers = []  # rendezvous wrappers for each queue type
        self.__render_target_ready = []  # semaphores
        self.__render_target_rendered = [] # semaphores
        self.__render_targets = []
        self.loaded_modules = {} # Cache of all loaded shader torch
        self.__frame_index = 0  # Current Frame
        self.__render_target_index = 0  # Current RT index in swapchain (not necessary the same of frame_index).
        self.__createInstance()
        self.__load_vk_calls()
        self.__createDebugInstance()
        self.__createSurface()
        self.__createPhysicalDevice()
        self.__createQueues()
        self.mem_properties = vkGetPhysicalDeviceMemoryProperties(self.__physical_device)
        self.memory_manager = VulkanMemoryManager(self.vk_device, self.__physical_device)
        self.__createSwapchain()

    def release(self):
        if self.vk_device is None:  # already released.
            return

        vkDeviceWaitIdle(self.vk_device)

        for p in self.__pipelines:  # Clean all pipelines hanging
            p._vk_destroy()

        for b in self.__resources:  # Clean all resources hanging
            b._vk_destroy()

        self.memory_manager.destroy()

        self.__queues = []  # list of queue objects for each queue family (only one queue used for family)
        self.__main_manager = None  # Main rendezvous queue used for presenting

        # release queue managers
        for m in self.__managers:
            m._vk_destroy()
        self.__managers = []  # rendezvous wrappers for each queue type

        # release render target image views
        self.__render_targets = []

        # semaphores
        [vkDestroySemaphore(self.vk_device, sem, None) for sem in self.__render_target_ready]
        [vkDestroySemaphore(self.vk_device, sem, None) for sem in self.__render_target_rendered]
        self.__render_target_ready = []
        self.__render_target_rendered = []

        # destroy shader module cache
        for k, v in self.loaded_modules.items():
            vkDestroyShaderModule(self.vk_device, v, None)
        self.loaded_modules = {}

        if self.__swapchain:
            self.vkDestroySwapchainKHR(self.vk_device, self.__swapchain, None)
        if self.vk_device:
            vkDestroyDevice(self.vk_device, None)
        if self.__callback:
            self.vkDestroyDebugReportCallbackEXT(self.__instance, self.__callback, None)
        if self.__surface:
            self.vkDestroySurfaceKHR(self.__instance, self.__surface, None)
        if self.__instance:
            vkDestroyInstance(self.__instance, None)

        self.vk_device = None  # Flag as released.

        gc.collect()

        print('[INFO] Destroyed vulkan instance')

    def set_debug_name(self, name: str):
        self.memory_manager.set_debug_name(name)

    @trace_destroying
    def __del__(self):
        self.release()

    def __load_vk_calls(self):
        self.vkCreateSwapchainKHR = vkGetInstanceProcAddr(self.__instance, 'vkCreateSwapchainKHR')
        self.vkGetSwapchainImagesKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetSwapchainImagesKHR')
        self.vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(
            self.__instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')
        self.vkCreateDebugReportCallbackEXT = vkGetInstanceProcAddr(
            self.__instance, "vkCreateDebugReportCallbackEXT")
        self.vkQueuePresentKHR = vkGetInstanceProcAddr(self.__instance, "vkQueuePresentKHR")
        self.vkAcquireNextImageKHR = vkGetInstanceProcAddr(self.__instance, "vkAcquireNextImageKHR")
        self.vkDestroyDebugReportCallbackEXT = vkGetInstanceProcAddr(self.__instance, "vkDestroyDebugReportCallbackEXT")
        self.vkDestroySwapchainKHR = vkGetInstanceProcAddr(self.__instance, "vkDestroySwapchainKHR")
        self.vkDestroySurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkDestroySurfaceKHR")

        self.vkCreateRayTracingPipelines = vkGetInstanceProcAddr(self.__instance, "vkCreateRayTracingPipelinesKHR")
        self.vkGetRayTracingShaderGroupHandles = \
            vkGetInstanceProcAddr(self.__instance, "vkGetRayTracingShaderGroupHandlesKHR")
        CommandBufferWrapper.vkCmdTraceRays = vkGetInstanceProcAddr(self.__instance, "vkCmdTraceRaysKHR")
        self.vkGetAccelerationStructureDeviceAddress = \
            vkGetInstanceProcAddr(self.__instance, "vkGetAccelerationStructureDeviceAddressKHR")
        self.vkCreateAccelerationStructureKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateAccelerationStructureKHR")
        self.vkGetAccelerationStructureBuildSizesKHR = vkGetInstanceProcAddr(
            self.__instance, "vkGetAccelerationStructureBuildSizesKHR")
        CommandBufferWrapper.vkCmdBuildAccelerationStructures = \
            vkGetInstanceProcAddr(self.__instance, "vkCmdBuildAccelerationStructuresKHR")
        CommandBufferWrapper.vkBuildAccelerationStructures = \
            vkGetInstanceProcAddr(self.__instance, "vkBuildAccelerationStructuresKHR")
        self.vkGetPhysicalDeviceProperties2=vkGetInstanceProcAddr(self.__instance, "vkGetPhysicalDeviceProperties2KHR")
        ResourceData.vkDestroyAccelerationStructure = vkGetInstanceProcAddr(self.__instance,
                                                                                 "vkDestroyAccelerationStructureKHR")
        self.vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = vkGetInstanceProcAddr(self.__instance,
                                                                                      "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV")
        # self.vkGetPhysicalDeviceFormatProperties = vkGetInstanceProcAddr(self.__instance, "vkGetPhysicalDeviceFormatProperties")
        if os.name == 'nt':
            self.vkGetMemoryWin32 = vkGetInstanceProcAddr(self.__instance, "vkGetMemoryWin32HandleKHR")
        else:
            self.vkGetMemoryFdKHR = vkGetInstanceProcAddr(self.__instance, "vkGetMemoryFdKHR")

    def __createSwapchain(self):
        if self.mode == 0:
            return  # No present at all
        self.__render_target_extent = VkExtent2D(self.width, self.height)
        if self.mode == 1:  # Offline rendezvous
            rt = self.create_image(
                VK_IMAGE_TYPE_2D,
                self.__render_target_format,
                0,
                VkExtent3D(self.width, self.height, 1),
                1, 1, False, VK_IMAGE_LAYOUT_UNDEFINED,
                self.render_target_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
            self.__render_targets = [rt]
        else:
            swapchain_create = VkSwapchainCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                flags=0,
                surface=self.__surface,
                minImageCount=2,
                imageFormat=self.__render_target_format,
                imageColorSpace=VK_COLORSPACE_SRGB_NONLINEAR_KHR,
                imageExtent=self.__render_target_extent,
                imageArrayLayers=1,
                imageUsage=self.render_target_usage,
                imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=0,
                pQueueFamilyIndices=None,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=VK_PRESENT_MODE_MAILBOX_KHR,
                clipped=VK_TRUE,
                oldSwapchain=None,
                preTransform=VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            self.__swapchain = self.vkCreateSwapchainKHR(self.vk_device, swapchain_create, None)
            images = self.vkGetSwapchainImagesKHR(self.vk_device, self.__swapchain)
            self.__render_targets = []
            self.__render_target_ready = []
            self.__render_target_rendered = []
            for img in images:
                # wrap swapchain image
                rt_data = ResourceData(self.vk_device, vk_description=VkImageCreateInfo(
                        imageType=VK_IMAGE_TYPE_2D,
                        mipLevels=1,
                        arrayLayers=1,
                        extent=VkExtent3D(self.width, self.height, 1),
                        format=self.__render_target_format
                    ),
                                       vk_memory_location=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                       vk_resource=img,
                                       is_buffer=False,
                                       w_memory=None,
                                       initial_state=ResourceState(
                        vk_access=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        vk_stage=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
                        queue_index=VK_QUEUE_FAMILY_IGNORED
                    ))
                rt = ResourceWrapper(rt_data)
                self.__render_targets.append(rt)
                # create semaphore for active render target
                self.__render_target_ready.append(vkCreateSemaphore(self.vk_device, VkSemaphoreCreateInfo(), None))
                self.__render_target_rendered.append(vkCreateSemaphore(self.vk_device, VkSemaphoreCreateInfo(), None))

            # Build command buffers for transitioning render target
            self.__presenting = vkAllocateCommandBuffers(
                self.vk_device,
                VkCommandBufferAllocateInfo(
                    commandPool=self.__main_manager.vk_pool,
                    level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=len(self.__render_targets)
                ))
            print(f"[INFO] Swapchain created with {len(images)} images...")

    def __createQueues(self):
        queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self.__physical_device)
        print("[INFO] %s available queue family" % len(queue_families))

        def get_queue_index(vk_bits):
            min_index = -1
            min_bits = 10000
            for i, queue_family in enumerate(queue_families):
                if queue_family.queueCount > 0 and (queue_family.queueFlags & vk_bits == vk_bits):
                    if min_bits > queue_family.queueFlags:
                        min_bits = queue_family.queueFlags
                        min_index = i
            return min_index

        # Preload queue families
        for bits in range(1, (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT) + 1):
            self.__queue_index[bits] = get_queue_index(bits)
        # Create a single queue for each family
        queues_create = [VkDeviceQueueCreateInfo(sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                 queueFamilyIndex=i,
                                                 queueCount=min(1, qf.queueCount),
                                                 pQueuePriorities=[1],
                                                 flags=0)
                         for i, qf in enumerate(queue_families)]

        existing_extensions = [e.extensionName for e in vkEnumerateDeviceExtensionProperties(self.__physical_device, None)]

        # Add here required extensions
        extensions = [
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            "VK_KHR_buffer_device_address",
            "VK_EXT_shader_atomic_float",
            "VK_KHR_shader_float16_int8",
            # "GLSL_EXT_ray_tracing",
            # "GLSL_EXT_ray_query",
            # "GLSL_EXT_ray_flags_primitive_culling",
            # "SPV_KHR_ray_tracing",
            # "SPV_KHR_ray_query"
        ]

        if os.name == 'nt':
            extensions += [
                "VK_KHR_external_memory_win32"
            ]
        else:
            extensions += [
                "VK_KHR_external_memory_fd"
            ]

        if self.support_raytracing:
            extensions += [
            # "VK_KHR_ray_query",   # Not supported in 1080, do not decomment
            "VK_KHR_deferred_host_operations",
            "VK_KHR_pipeline_library",
            "VK_KHR_acceleration_structure",
            "VK_KHR_ray_tracing_pipeline",
            ]
        if self.support_cooperative_matrices:
            extensions += [
                "VK_NV_cooperative_matrix",
            ]

        not_present_extensions = [e for e in extensions if e not in existing_extensions]

        dev_features = self.__physical_device_features

        storage_features = VkPhysicalDevice16BitStorageFeatures(
            # storageBuffer16BitAccess=self.support_cooperative_matrices,
        )

        coop_features = VkPhysicalDeviceCooperativeMatrixFeaturesNV(
            pNext = storage_features,
            # cooperativeMatrix=self.support_cooperative_matrices
        )

        atom_features =  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT(
            pNext=coop_features,
            shaderBufferFloat32Atomics=True,
            shaderBufferFloat32AtomicAdd=True,
        )

        rob_features = VkPhysicalDeviceRobustness2FeaturesEXT(
            pNext = atom_features,
            nullDescriptor=True,
        )

        ads_features = VkPhysicalDeviceAccelerationStructureFeaturesKHR(
            pNext = rob_features,
            accelerationStructure=self.support_raytracing,
            # accelerationStructureHostCommands=True,
            # descriptorBindingAccelerationStructureUpdateAfterBind=True,
        )

        rt_features = VkPhysicalDeviceRayTracingPipelineFeaturesKHR(
            pNext = ads_features,
            rayTracingPipeline=self.support_raytracing,
        )

        features = VkPhysicalDeviceVulkan12Features(
            pNext=rt_features,

            bufferDeviceAddress=True,
            bufferDeviceAddressCaptureReplay=True,
            bufferDeviceAddressMultiDevice=True,
            scalarBlockLayout=True,
            shaderSampledImageArrayNonUniformIndexing=True,
            runtimeDescriptorArray=True,
            descriptorBindingVariableDescriptorCount=True,
            descriptorBindingPartiallyBound=True,
            # descriptorBindingStorageImageUpdateAfterBind=True,
            # descriptorBindingStorageBufferUpdateAfterBind=True,
            # descriptorBindingUniformBufferUpdateAfterBind=True,
            # descriptorBindingStorageTexelBufferUpdateAfterBind=True,
            # descriptorBindingUniformTexelBufferUpdateAfterBind=True,
            # descriptorBindingUpdateUnusedWhilePending=True,
            # descriptorBindingSampledImageUpdateAfterBind=True,
            # shaderFloat16=True,
            # shaderSubgroupExtendedTypes=True,
            # vulkanMemoryModel=True,
            # vulkanMemoryModelDeviceScope=True
        )

        device_create = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=VkPhysicalDeviceFeatures2(pNext=features, features=dev_features),
            pQueueCreateInfos=queues_create,
            queueCreateInfoCount=len(queues_create),
            # pEnabledFeatures=self.__physical_device_features,
            flags=0,
            enabledLayerCount=len(self.__layers),
            ppEnabledLayerNames=self.__layers,
            # enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        self.vk_device = vkCreateDevice(self.__physical_device, device_create, None)

        # load calls
        self.vkGetBufferDeviceAddress = vkGetDeviceProcAddr(self.vk_device, "vkGetBufferDeviceAddressKHR")


        self.__queues = [None if qf.queueCount == 0 else vkGetDeviceQueue(
            device=self.vk_device,
            queueFamilyIndex=i,
            queueIndex=0) for i, qf in enumerate(queue_families)]

        # resolve command list manager types for each queue
        self.__managers = []
        for i, qf in enumerate(queue_families):
            pool = vkCreateCommandPool(
                device=self.vk_device,
                pCreateInfo=VkCommandPoolCreateInfo(
                    queueFamilyIndex=i,
                    flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
                ),
                pAllocator=None)
            self.__managers.append(CommandPoolWrapper(self, self.__queues[i], pool, i))

        for i, qf in enumerate(queue_families):
            support_present = self.mode <= 1 or self.vkGetPhysicalDeviceSurfaceSupportKHR(
                physicalDevice=self.__physical_device,
                queueFamilyIndex=i,
                surface=self.__surface)
            if qf.queueCount > 0 and (qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) and support_present:
                self.__main_manager = self.__managers[i]
                self.__main_queue_index = i
        print("[INFO] Device and queues created...")

    def __createSurface(self):
        if self.mode == 2:  # SDL Window
            """
            SDL2 Initialization for windows support
            """
            import os
            os.environ["PYSDL2_DLL_PATH"] = "./third-party/SDL2"
            import sdl2
            import sdl2.ext

            if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
                raise Exception(sdl2.SDL_GetError())

            window = sdl2.SDL_CreateWindow(
                'Vulkan Window'.encode('ascii'),
                sdl2.SDL_WINDOWPOS_UNDEFINED,
                sdl2.SDL_WINDOWPOS_UNDEFINED, self.width, self.height, 0)

            wm_info = sdl2.SDL_SysWMinfo()
            sdl2.SDL_VERSION(wm_info.version)
            sdl2.SDL_GetWindowWMInfo(window, ctypes.byref(wm_info))

            if wm_info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
                self.__extensions.append('VK_KHR_win32_surface')
            elif wm_info.subsystem == sdl2.SDL_SYSWM_X11:
                self.__extensions.append('VK_KHR_xlib_surface')
            elif wm_info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
                self.__extensions.append('VK_KHR_wayland_surface')
            else:
                raise Exception("Platform not supported")

            class SDLWindow(WindowWrapper):
                def __init__(self):
                    self.event = sdl2.SDL_Event()

                def poll_events(self) -> (Event, object):
                    if sdl2.SDL_PollEvent(ctypes.byref(self.event)) != 0:
                        if self.event.type == sdl2.SDL_QUIT:
                            return Event.CLOSED, None
                    return Event.NONE, None

            self.__window = SDLWindow()

            def surface_xlib():
                print("Create Xlib surface")
                vkCreateXlibSurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateXlibSurfaceKHR")
                surface_create = VkXlibSurfaceCreateInfoKHR(
                    sType=VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    dpy=wm_info.info.x11.display,
                    window=wm_info.info.x11.window,
                    flags=0)
                return vkCreateXlibSurfaceKHR(self.__instance, surface_create, None)

            def surface_wayland():
                print("Create wayland surface")
                vkCreateWaylandSurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateWaylandSurfaceKHR")
                surface_create = VkWaylandSurfaceCreateInfoKHR(
                    sType=VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                    display=wm_info.info.wl.display,
                    surface=wm_info.info.wl.surface,
                    flags=0)
                return vkCreateWaylandSurfaceKHR(self.__instance, surface_create, None)

            def surface_win32():
                def get_instance(hWnd):
                    """Hack needed before SDL 2.0.6"""
                    from cffi import FFI
                    _ffi = FFI()
                    _ffi.cdef('long __stdcall GetWindowLongA(void* hWnd, int nIndex);')
                    _lib = _ffi.dlopen('User32.dll')
                    return _lib.GetWindowLongA(_ffi.cast('void*', hWnd), -6)

                print("[INFO] Windows surface created...")
                vkCreateWin32SurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateWin32SurfaceKHR")
                surface_create = VkWin32SurfaceCreateInfoKHR(
                    sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    hinstance=get_instance(wm_info.info.win.window),
                    hwnd=wm_info.info.win.window,
                    flags=0)
                return vkCreateWin32SurfaceKHR(self.__instance, surface_create, None)

            surface_mapping = {
                sdl2.SDL_SYSWM_X11: surface_xlib,
                sdl2.SDL_SYSWM_WAYLAND: surface_wayland,
                sdl2.SDL_SYSWM_WINDOWS: surface_win32
            }

            self.__surface = surface_mapping[wm_info.subsystem]()

    def __createDebugInstance(self):
        def debug_callback(*args):
            print('DEBUG: ' + args[5] + ' ' + args[6])
            return 0

        debug_create = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debug_callback)
        self.__callback = self.vkCreateDebugReportCallbackEXT(self.__instance, debug_create, None)
        print('[INFO] Debug instance created...')

    def __createInstance(self):
        self.__layers = [l.layerName for l in vkEnumerateInstanceLayerProperties()]
        if 'VK_LAYER_KHRONOS_validation' in self.__layers:
            self.__layers = ['VK_LAYER_KHRONOS_validation']
        if 'VK_LAYER_LUNARG_standard_validation' in self.__layers:
            self.__layers = ['VK_LAYER_LUNARG_standard_validation']
        if self.enable_validation_layers and len(self.__layers) == 0:
            raise Exception("validation layers requested, but not layer available!")
        self.__extensions = [e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)]
        if self.enable_validation_layers:
            self.__extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
        appInfo = VkApplicationInfo(
            # sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Rendering with Python VK',
            applicationVersion=VK_MAKE_VERSION(1, 2, 0),
            pEngineName='pyvulkan',
            engineVersion=VK_MAKE_VERSION(1, 2, 0),
            apiVersion=VK_MAKE_VERSION(1, 2, 0)
        )
        if self.enable_validation_layers:
            instanceInfo = VkInstanceCreateInfo(
                pApplicationInfo=appInfo,
                enabledLayerCount=len(self.__layers),
                ppEnabledLayerNames=self.__layers,
                enabledExtensionCount=len(self.__extensions),
                ppEnabledExtensionNames=self.__extensions
            )
        else:
            instanceInfo = VkInstanceCreateInfo(
                pApplicationInfo=appInfo,
                enabledLayerCount=0,
                enabledExtensionCount=len(self.__extensions),
                ppEnabledExtensionNames=self.__extensions
            )

        self.__instance = vkCreateInstance(instanceInfo, None)
        print("[INFO] Vulkan Instance created...")

    def __createPhysicalDevice(self):
        selected_device = 0
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            selected_device = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])

        all_devices = list(vkEnumeratePhysicalDevices(self.__instance))
        self.__physical_device = all_devices[selected_device]
        self.__physical_device_features = vkGetPhysicalDeviceFeatures(self.__physical_device)
        physical_device_properties = vkGetPhysicalDeviceProperties(self.__physical_device)
        coop_prop=VkPhysicalDeviceCooperativeMatrixPropertiesNV()
        ads_prop=VkPhysicalDeviceAccelerationStructurePropertiesKHR(pNext=coop_prop)
        rt_prop = VkPhysicalDeviceRayTracingPipelinePropertiesKHR(pNext=ads_prop)
        vk12_prop = VkPhysicalDeviceVulkan12Properties(pNext=rt_prop)
        prop = VkPhysicalDeviceProperties2(pNext=vk12_prop)
        self.vkGetPhysicalDeviceProperties2(self.__physical_device, prop)

        self.support_cooperative_matrices = coop_prop.cooperativeMatrixSupportedStages != 0
        self.support_raytracing = ads_prop.maxGeometryCount > 0

        coop_properties = self.vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(self.__physical_device)

        formats = [VK_FORMAT_R32G32B32_SFLOAT]
        for f in formats:
            format_prop = vkGetPhysicalDeviceFormatProperties(self.__physical_device, f)

        self.raytracing_properties = rt_prop
        major, minor = VK_VERSION_MAJOR(prop.properties.apiVersion), VK_VERSION_MINOR(prop.properties.apiVersion)
        print("[INFO] Available devices: %s" % len(all_devices))
        print("[INFO] Selected device: %s\n" % physical_device_properties.deviceName)
        print("[INFO] Selected device index: %s\n" % selected_device)

    def get_window(self):
        return self.__window

    def begin_frame(self):
        """
        Peek a render target from swap chain and set as current
        """
        if self.mode == 1:  # Offline rendezvous
            return

        self.__render_target_index = self.vkAcquireNextImageKHR(
            self.vk_device,
            self.__swapchain,
            1000000000,
            self.__render_target_ready[self.__frame_index],
            VK_NULL_HANDLE)

    def get_render_target(self, index):
        return self.__render_targets[index]

    def get_render_target_index(self):
        return self.__render_target_index

    def get_number_of_frames(self):
        return max(1, len(self.__render_targets))

    def get_frame_index(self):
        return self.__frame_index

    def flush_pending_and_wait(self):
        for m in self.__managers:
            m.flush()

    def end_frame(self):
        self.flush_pending_and_wait()
        if self.mode == 1:  # offline
            return

        # Finish transitioning the render target to present
        rt: ResourceWrapper = self.get_render_target(self.get_render_target_index())
        cmdList = self.__presenting[self.__render_target_index]  # cmdList for transitioning rt
        vkResetCommandBuffer(commandBuffer=cmdList, flags=VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)
        vkBeginCommandBuffer(cmdList, VkCommandBufferBeginInfo())
        rt.add_barrier(cmdList, ResourceState(
            vk_access=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
            vk_stage=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            vk_layout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            queue_index=VK_QUEUE_FAMILY_IGNORED
        ))
        vkEndCommandBuffer(cmdList)
        # Wait for completation
        submit_create = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmdList],
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.__render_target_ready[self.__frame_index]],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.__render_target_rendered[self.__frame_index]])

        vkQueueSubmit(self.__main_manager.vk_queue, 1, [submit_create], None)
        # Present render target
        present_create = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.__render_target_rendered[self.__frame_index]],
            swapchainCount=1,
            pSwapchains=[self.__swapchain],
            pImageIndices=[self.__render_target_index],
            pResults=None)
        self.vkQueuePresentKHR(self.__main_manager.vk_queue, present_create)
        if self.enable_validation_layers:
            vkQueueWaitIdle(self.__main_manager.vk_queue)
        # update frame index
        self.__frame_index = (self.__frame_index + 1) % len(self.__render_targets)

    def create_cmdList(self, queue_bits):
        return self.__managers[self.__queue_index[queue_bits]].get_cmdList()

    def __findMemoryType(self, filter, properties):
        for i, prop in enumerate(self.mem_properties.memoryTypes):
            if (filter & (1 << i)) and ((prop.propertyFlags & properties) == properties):
                return i
        raise Exception("failed to find suitable memory type!")

    def __resolve_initial_state(self, is_buffer, is_ads, usage, properties):
        if is_ads:
            return ResourceState(
                vk_access=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                vk_stage=VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
                queue_index=VK_QUEUE_FAMILY_IGNORED
            )
        return ResourceState(
            vk_access=0,
            vk_stage=VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
            queue_index=VK_QUEUE_FAMILY_IGNORED
        )

    def collect(self):
        self.memory_manager.collect()

    def clear_cache(self):
        self.memory_manager.clear_cache()

    def create_buffer_data(self, size, usage, properties, is_ads = False):
        prev = VkExternalMemoryBufferCreateInfo(
            handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT if os.name == 'nt' else VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        )
        info = VkBufferCreateInfo(
            pNext=prev,
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        buffer = vkCreateBuffer(self.vk_device, info, None)
        w_memory : VulkanMemory = self.memory_manager.allocate_memory_for_buffer(buffer, properties)
        vkBindBufferMemory(self.vk_device, buffer, w_memory.get_vulkan_memory(), w_memory.get_vulkan_memory_offset())
        resource_data = ResourceData(self.vk_device, info, properties, buffer, w_memory, True,
                               self.__resolve_initial_state(True, is_ads, usage, properties))
        self.__resources.add(resource_data)
        return resource_data

    def create_buffer(self, size, usage, properties, is_ads: bool = False):
        return ResourceWrapper(resource_data=self.create_buffer_data(size, usage, properties, is_ads))

    def _get_device_address(self, w_resource: ResourceWrapper):
        if w_resource is None:
            return None
        add = VkDeviceOrHostAddressKHR()
        #if w_resource.resource_data.is_gpu:
        add.deviceAddress = w_resource.get_device_ptr()
        #else:
        #    add.hostAddress = w_resource.get_host_ptr()
        return add

    def _get_device_address_const(self, w_resource: ResourceWrapper):
        if w_resource is None:
            return None
        add = VkDeviceOrHostAddressConstKHR()
        #if w_resource.resource_data.is_gpu:
        add.deviceAddress = w_resource.get_device_ptr()
        #else:
        #    add.hostAddress = w_resource.get_host_ptr()
        return add

    def _resolve_description(self, geometry_type, element_description):
        if geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR:
            v, v_stride, i, t = element_description
            max_vertex = v.get_size() // v_stride
            data = VkAccelerationStructureGeometryKHR(
                geometryType=geometry_type,
                geometry=VkAccelerationStructureGeometryDataKHR(
                triangles= VkAccelerationStructureGeometryTrianglesDataKHR(
                vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
                vertexData=self._get_device_address_const(v),
                vertexStride=v_stride,
                maxVertex=max_vertex,
                indexType=VK_INDEX_TYPE_UINT32,
                indexData=self._get_device_address_const(i),
                transformData=self._get_device_address_const(t)
            )))
            if i:
                primitives = i.get_size() // 4 // 3
            else:
                primitives = v.get_size() // v_stride // 3
        elif geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR:
            aabbs = element_description
            data = VkAccelerationStructureGeometryKHR(
                geometryType=geometry_type,
                geometry=VkAccelerationStructureGeometryDataKHR(
                aabbs=VkAccelerationStructureGeometryAabbsDataKHR(
                stride=24,
                data=self._get_device_address_const(aabbs)
            )))
            primitives = aabbs.get_size() // 24
        else:
            instances = element_description
            data = VkAccelerationStructureGeometryKHR(
                geometryType=geometry_type,
                geometry=VkAccelerationStructureGeometryDataKHR(
                instances=VkAccelerationStructureGeometryInstancesDataKHR(
                data=self._get_device_address_const(instances),
                arrayOfPointers=False
            )))
            primitives = instances.get_size() // 64

        range = VkAccelerationStructureBuildRangeInfoKHR(
            primitiveCount=primitives,
            primitiveOffset=0,
            transformOffset=0,
            firstVertex=0
        )
        return data, range

    def create_ads(self, geometry_type=0, descriptions=None):
        """
        for triangles a description is a tuple in the form (v, v_stride, i, t). From there a rangeinfo can be extracted
        for aabbs a decsription is directly the aabb buffer
        for instances a description is directly the instance buffer
        """
        # Compute the required size of the buffer and the scratch buffer
        if descriptions is None:
            descriptions = []
        structure_type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR if geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR else VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        datas, ranges = zip(*[list(self._resolve_description(geometry_type, d)) for d in descriptions])
        info = VkAccelerationStructureBuildGeometryInfoKHR(
            type=structure_type,
            mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            geometryCount=len(descriptions),
            pGeometries=datas
        )
        sizes = VkAccelerationStructureBuildSizesInfoKHR()
        self.vkGetAccelerationStructureBuildSizesKHR(self.vk_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                     info, [range.primitiveCount for range in ranges], sizes)
        # Create a buffer to store the ads
        ads_buffer = self.create_buffer(sizes.accelerationStructureSize * 2,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, False)
        # Create object
        create_info = VkAccelerationStructureCreateInfoKHR(
            buffer=ads_buffer.resource_data.vk_resource,
            offset=0, size=sizes.accelerationStructureSize,
            type=structure_type
        )
        ads = self.vkCreateAccelerationStructureKHR(self.vk_device, create_info, None)
        ads_buffer.resource_data.bind_ads(ads)
        query_device_address_info = VkAccelerationStructureDeviceAddressInfoKHR(accelerationStructure=ads)
        device_address = self.vkGetAccelerationStructureDeviceAddress(self.vk_device, query_device_address_info)
        return ads_buffer, info, ranges, device_address, max(sizes.buildScratchSize, sizes.updateScratchSize)*2

    def create_image_data(self, image_type, image_format, flags, extent, mips, layers, linear,
                     initial_layout, usage, properties):
        external_info = VkExternalMemoryImageCreateInfo(
            handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT if os.name == 'nt' else VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        )
        info = VkImageCreateInfo(
            pNext=external_info,
            imageType=image_type,
            format=image_format,
            flags=flags,
            extent=extent,
            arrayLayers=layers,
            mipLevels=mips,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_LINEAR if linear else VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=initial_layout)
        image = vkCreateImage(self.vk_device, info, None)
        w_memory : VulkanMemory = self.memory_manager.allocate_memory_for_image(image, properties)
        vkBindImageMemory(self.vk_device, image, w_memory.get_vulkan_memory(), w_memory.get_vulkan_memory_offset())
        resource_data = ResourceData(self.vk_device, info, properties, image, w_memory, False,
                                     self.__resolve_initial_state(False, False, usage, properties))
        self.__resources.add(resource_data)
        return resource_data

    def create_image(self, image_type, image_format, flags, extent, mips, layers, linear,
                     initial_layout, usage, properties):
        resource_data = self.create_image_data(image_type, image_format, flags, extent, mips, layers, linear, initial_layout, usage, properties)
        return ResourceWrapper(resource_data)

    def create_sampler(self, mag_filter, min_filter, mipmap_mode, address_U, address_V, address_W,
                       mip_LOD_bias, enable_anisotropy, max_anisotropy, enable_compare,
                       compare_op, min_LOD, max_LOD, border_color, use_unnormalized_coordinates
                       ):
        info = VkSamplerCreateInfo(
            magFilter=mag_filter,
            minFilter=min_filter,
            mipmapMode=mipmap_mode,
            addressModeU=address_U,
            addressModeV=address_V,
            addressModeW=address_W,
            mipLodBias=mip_LOD_bias,
            anisotropyEnable=enable_anisotropy,
            maxAnisotropy=max_anisotropy,
            compareEnable=enable_compare,
            compareOp=compare_op,
            minLod=min_LOD,
            maxLod=max_LOD,
            borderColor=border_color,
            unnormalizedCoordinates=use_unnormalized_coordinates
        )
        return SamplerWrapper(self, vkCreateSampler(self.vk_device, info, None))

    def create_pipeline(self, vk_pipeline_type):
        p = PipelineBindingWrapper(w_device=self, pipeline_type=vk_pipeline_type)
        self.__pipelines.add(p)
        return p

