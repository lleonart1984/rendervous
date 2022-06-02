import weakref
import math
import sys
import os
import vulkan as vk
from cuda import cudart
from cuda import cuda
import ctypes
import numpy as np
import torch
import cupy as cp
import threading


__SHOW_ALLOCATE_AND_FREES__ = False


class _Node:
    def __init__(self, value, prev, next):
        self.value = value
        self.prev = None if prev is None else weakref.ref(prev)
        self.next = next

    def remove(self):
        p = self.prev()
        next = self.next
        p.next = next
        next.prev = weakref.ref(p)

    def insert_after(self, value):
        node = _Node(value, self, self.next)
        next = self.next
        self.next = node
        if next is not None:
            next.prev = weakref.ref(node)
        return node

    def insert_before(self, value):
        node = _Node(value, self.prev(), self)
        p = self.prev()
        self.prev = weakref.ref(node)
        p.next = node
        return node


class _Block:
    def __init__(self, offset, size, occupied):
        self.offset = offset
        self.size = size
        self.occupied = occupied


def _level_index_of(size):
    if size <= 1:
        return int(size)
    return int(math.log2(size - 1) + 2)


def _align(x: int, alignment):
    return (x + alignment - 1) // alignment * alignment


class Allocator:

    def __init__(self, capacity):
        self.capacity = capacity
        self._start_node = _Node(None, None, None)
        self._end_node = self._start_node.insert_after(None)
        n_b = self._start_node.insert_after(_Block(0, capacity, False))
        self._free_blocks = [[] for s in range(_level_index_of(capacity)+1)]  # for specific size gives a list of free blocks
        self._occupied_blocks = {}  # offset to node
        self._free_blocks[-1].append(n_b)
        self.locker = threading.Lock()

    def malloc(self, size: int, alignment: int = 16) -> int:
        with self.locker:
            size = size + alignment - 1  # grant to find a suitable alignable offset
            start_index = _level_index_of(size)
            for l in range(start_index, len(self._free_blocks)):
                if len(self._free_blocks[l]) > 0:  # Found a block
                    node = self._free_blocks[l].pop()
                    b : _Block = node.value
                    assert b.size >= size, "Wrong implementation. Block was find in a level doesnt correspond with the size"
                    new_free_block = _Block(b.offset+size, b.size-size, False)
                    b.size = size  #shrink block to allocated size
                    b.occupied = True
                    if new_free_block.size > 0: # left something free
                        new_free_node = node.insert_after(new_free_block)
                        self._free_blocks[_level_index_of(new_free_block.size)].append(new_free_node)
                    aligned_offset = _align(b.offset, alignment)
                    self._occupied_blocks[aligned_offset] = node
                    return aligned_offset
        raise Exception("Out of Memory, can not fit the allocation size")

    def free(self, ptr: int):
        # print(f'freeing ptr: {ptr}')
        with self.locker:
            assert ptr in self._occupied_blocks, "The memory address was not allocated in this allocator or was already free"
            node_to_free = self._occupied_blocks.pop(ptr)
            node_to_free.value.occupied = False
            prev = node_to_free.prev()
            if prev is not self._start_node and not prev.value.occupied:
                # merge with previous
                node_to_free.value.offset = prev.value.offset
                node_to_free.value.size += prev.value.size
                l = self._free_blocks[_level_index_of(prev.value.size)]
                if prev in l:
                    # assert prev in l, f"Node {prev} is not in list {l}"
                    l.remove(prev)
                prev.remove()
            next = node_to_free.next
            if next is not self._end_node and not next.value.occupied:
                node_to_free.value.size += next.value.size
                l = self._free_blocks[_level_index_of(next.value.size)]
                if next in l:
                    # assert next in l, f"Node {next} is not in list {l}"
                    l.remove(next)
                next.remove()
            self._free_blocks[_level_index_of(node_to_free.value.size)].append(node_to_free)

    def is_empty(self):
        # Has only one node as is not occupied
        return self._start_node.next.next is self._end_node and not self._start_node.next.value.occupied


class VulkanMemoryPage:
    def __init__(self, vk_device, capacity, memory_index: int, is_cpu: bool, is_gpu: bool):
        self.vk_device = vk_device
        self.capacity = capacity
        self.is_cpu = is_cpu
        self.is_gpu = is_gpu
        self.allocator = Allocator(capacity)
        self.vk_memory = None  # TODO: Create memory heap here and retrieve ptr
        self.memory_cpu_ptr = None
        self.memory_cuda_ptr = None
        self.memory_cuda_mem = None
        self.memory_vk_handle = None
        self.memory_device_ptr = None  # Memory managed by vulkan
        self.memory_as_tensor = None
        self.memory_cpu_mapped_ptr = None
        prev = None
        if self.is_gpu:  # Prepare memory for exporting handle
            if os.name == 'nt':  # security for win32
                import win32security
                sec_desc = win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
                    "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)", 1)
                pdesc = vk.ffi.from_buffer(sec_desc)
                prev = vk.VkExportMemoryWin32HandleInfoKHR(
                    pNext=prev,
                    pAttributes=[(24, pdesc, 1)],
                    dwAccess=0x80000000 | 1,
                    name=None
                )
            prev = vk.VkExportMemoryAllocateInfo(
                pNext=prev,
                handleTypes=
                vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                if os.name == 'nt'
                else vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
            )
        prev = vk.VkMemoryAllocateFlagsInfo(  # Allowing to get device address
            pNext=prev,
            flags=vk.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT  # Mandatory feature in rendezvous
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            pNext=prev,
            allocationSize=capacity,
            memoryTypeIndex=memory_index
        )
        self.vk_memory = vk.vkAllocateMemory(self.vk_device, alloc_info, None)  # Allocate vulkan memory

        # Create a buffer with the full block to get mapped address (cpu) or gpu device address (gpu)
        external_info = vk.VkExternalMemoryBufferCreateInfo(handleTypes=
                                            vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                                            if os.name == 'nt'
                                            else vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                                            )
        full_block = vk.vkCreateBuffer(self.vk_device, vk.VkBufferCreateInfo(pNext=external_info, flags=0, size=capacity,
                                                     usage=vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT), None)
        vk.vkBindBufferMemory(self.vk_device, full_block, self.vk_memory, 0)
        vkGetBufferDeviceAddressKHR = vk.vkGetDeviceProcAddr(self.vk_device, "vkGetBufferDeviceAddressKHR")
        buffer_info = vk.VkBufferDeviceAddressInfo(buffer=full_block)
        self.memory_device_ptr = vkGetBufferDeviceAddressKHR(self.vk_device, buffer_info)
        vk.vkDestroyBuffer(self.vk_device, full_block, None)

        # create a full tensor to map all memory in any case
        if is_cpu:
            # Permanent mapping for cpu based pages.
            self.memory_cpu_mapped_ptr = vk.vkMapMemory(self.vk_device, self.vk_memory, 0, capacity, 0)
            self.memory_cpu_ptr = np.frombuffer(self.memory_cpu_mapped_ptr).__array_interface__['data'][0]
            self.memory_cuda_ptr = self.memory_cpu_ptr  # if cpu let cuda ptr to be the cpu ptr.
            self.memory_as_tensor = torch.from_numpy(np.frombuffer(self.memory_cpu_mapped_ptr, dtype=np.uint8))

        if is_gpu:
            if os.name == 'nt':
                vkmemwin32info = vk.VkMemoryGetWin32HandleInfoKHR(memory=self.vk_memory,
                                                               handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)
                vkGetMemoryWin32 = vk.vkGetDeviceProcAddr(self.vk_device, "vkGetMemoryWin32HandleKHR")
                self.memory_vk_handle = vkGetMemoryWin32(self.vk_device, vkmemwin32info)
            else:
                vkmemfdinfo = vk.VkMemoryGetFdInfoKHR(memory=self.vk_memory,
                                                   handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
                vkGetMemoryFdKHR = vk.vkGetDeviceProcAddr(self.vk_device, "vkGetMemoryFdKHR")
                self.memory_vk_handle = vkGetMemoryFdKHR(self.vk_device, vkmemfdinfo)
            external_mem_hd = cudart.cudaExternalMemoryHandleDesc(0)
            external_mem_hd.size = capacity
            if os.name == 'nt':
                external_mem_hd.handle.win32.handle = vk.ffi.buffer(self.memory_vk_handle, 8)
                external_mem_hd.type = cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32
            else:
                external_mem_hd.handle.fd = self.memory_vk_handle
                external_mem_hd.type = cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd
            r, self.memory_cuda_mem = cudart.cudaImportExternalMemory(external_mem_hd)
            external_mem_buffer_desc = cudart.cudaExternalMemoryBufferDesc(0)
            external_mem_buffer_desc.offset = 0
            external_mem_buffer_desc.flags = 0
            external_mem_buffer_desc.size = capacity
            r, self.memory_cuda_ptr = cudart.cudaExternalMemoryGetMappedBuffer(self.memory_cuda_mem, external_mem_buffer_desc)
            cp_array = cp.ndarray(
                shape=(capacity,),
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.memory_cuda_ptr, capacity, self, 0), 0)
            )
            self.memory_as_tensor = torch.as_tensor(cp_array, dtype=torch.uint8, device=torch.device('cuda:0'))
            base = self.memory_as_tensor if self.memory_as_tensor._base is None else self.memory_as_tensor._base
            base.is_vulkanized = True  # Mark the tensor as vulkanized if is on the gpu

    def malloc(self, size: int, alignment: int = 16):
        offset = self.allocator.malloc(size, alignment)
        return offset

    def free(self, offset: int):
        self.allocator.free(offset)

    def device_ptr(self):
        return self.memory_device_ptr

    def cuda_ptr(self):
        return self.memory_cuda_ptr

    def host_ptr(self):
        return vk.ffi.from_buffer(self.memory_cpu_mapped_ptr)

    def destroy(self):
        if self.vk_memory is None:
            return
        if self.is_cpu:
            vk.vkUnmapMemory(self.vk_device, self.vk_memory)
        if self.is_gpu:
            if os.name == 'nt':
                if self.memory_vk_handle is not None and vk.ffi.NULL != self.memory_vk_handle:
                    ctypes.windll.kernel32.CloseHandle(int(vk.ffi.cast('long', self.memory_vk_handle)))
                    # win32api.CloseHandle(int(ffi.cast('long', vk_handle)))
            else:
                pass
                # import posix
                # posix.r
                # posix.close(vk_handle)
            r = cudart.cudaFree(self.memory_cuda_ptr)
            r = cuda.cuDestroyExternalMemory(self.memory_cuda_mem)
        vk.vkFreeMemory(self.vk_device, self.vk_memory, None)
        self.vk_memory = None  # TODO: Create memory heap here and retrieve ptr
        self.memory_cpu_ptr = None
        self.memory_cuda_ptr = None
        self.memory_cuda_mem = None
        self.memory_vk_handle = None
        self.memory_device_ptr = None
        self.memory_as_tensor = None
        self.memory_cpu_mapped_ptr = None

    def __del__(self):
        self.destroy()


class VulkanMemory:
    def __init__(self, page_allocator: VulkanMemoryPage, offset, size):
        self.page_allocator = page_allocator
        self.offset = offset
        self.size = size
        self.debug_name = None

    def is_gpu(self):
        return self.page_allocator.is_gpu

    def is_cpu(self):
        return self.page_allocator.is_cpu

    def device_ptr(self):
        return self.page_allocator.device_ptr() + self.offset

    def cuda_ptr(self):
        return self.page_allocator.cuda_ptr() + self.offset

    def host_ptr(self):
        return self.page_allocator.host_ptr() + self.offset

    def get_vulkan_memory(self):
        return self.page_allocator.vk_memory

    def get_vulkan_memory_offset(self):
        return self.offset

    def get_tensor(self):
        return self.page_allocator.memory_as_tensor[self.offset: self.offset + self.size]

    def free(self):
        if self.page_allocator is None:
            return
        p = self.page_allocator
        self.page_allocator = None
        p.free(self.offset)
        self.offset = 0
        self.size = 0
        if __SHOW_ALLOCATE_AND_FREES__:
            print('freeing memory of '+('anonym resource' if self.debug_name is None else self.debug_name))

    # def __del__(self):
    #     self.free()


class VulkanAllocator:

    def __init__(self, vk_device, memory_index: int, memory_properties):
        self.pages = []
        self.vk_device = vk_device
        self.is_cpu = memory_properties & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        self.is_gpu = memory_properties & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        self.is_gpu_dedicated = self.is_gpu and not self.is_cpu
        self.reserved_memory = 0
        self.memory_index = memory_index
        self._allocated_objects = weakref.WeakSet()

    def allocate(self, size: int, alignment: int) -> VulkanMemory:
        for p in self.pages:
            try:
                offset = p.malloc(size, alignment)
                m = VulkanMemory(p, offset, size)
                self._allocated_objects.add(m)
                return m
            except:
                pass
        # Here a new page is required
        # 64MB minimum for host memory or 1GB minimum on the GPU
        page_capacity = max(1024**3 if self.is_gpu_dedicated else 64*1024**2, 2 ** int(math.log2(size+alignment) + 1))
        print(f"[INFO] Creating page with {page_capacity//(1 << 20)}MB")
        page = VulkanMemoryPage(self.vk_device, page_capacity, self.memory_index, self.is_cpu, self.is_gpu)
        self.reserved_memory += page_capacity
        self.pages.append(page)
        offset = page.malloc(size, alignment)
        m = VulkanMemory(page, offset, size)
        self._allocated_objects.add(m)
        return m

    def collect(self):
        for mem in self._allocated_objects:
            if sys.getrefcount(mem) == 1:
                print('freeing memory in collect')
                mem.free()

    def clear_cache(self):
        """
        Free memory of empty pages
        """
        for i, p in enumerate(self.pages):
            if p.is_empty():
                p.destroy()
                self.pages[i] = None
        self.pages = [p for p in self.pages if p is not None]

    def destroy(self):
        if self.pages is None:
            return
        for p in self.pages:
            p.destroy()
        self.pages = None


class VulkanMemoryManager:

    def __init__(self, vk_device, vk_physical_device):
        self.vk_device = vk_device
        self.mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(vk_physical_device)
        self.allocators = { } # map from memory index to VulkanAllocator
        self.debug_name = None

    def __findMemoryType(self, filter, mem_properties):
        for i, prop in enumerate(self.mem_properties.memoryTypes):
            if (filter & (1 << i)) and ((prop.propertyFlags & mem_properties) == mem_properties):
                return i
        raise Exception("failed to find suitable memory type!")

    def collect(self):
        for k, v in self.allocators.items():
            v.collect()

    def clear_cache(self):
        for k, v in self.allocators.items():
            v.clear_cache()

    def set_debug_name(self, memory_name):
        self.debug_name = memory_name

    def peek_debug_name(self, mem):
        if self.debug_name is None:
            return 'Memory '+str(mem.size)
        d = self.debug_name
        self.debug_name = None
        return d

    def allocate_memory_for_buffer(self, buffer, memory_location):
        self.collect()
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.vk_device, buffer)
        index = self.__findMemoryType(mem_reqs.memoryTypeBits, memory_location)
        if index not in self.allocators:
            self.allocators[index] = VulkanAllocator(self.vk_device, index, self.mem_properties.memoryTypes[index].propertyFlags)
        mem = self.allocators[index].allocate(mem_reqs.size, mem_reqs.alignment)
        mem.debug_name = self.peek_debug_name(mem)
        if __SHOW_ALLOCATE_AND_FREES__:
            print(f'creating memory for {mem.debug_name}')
        return mem

    def allocate_memory_for_image(self, image, memory_location):
        mem_reqs = vk.vkGetImageMemoryRequirements(self.vk_device, image)
        index = self.__findMemoryType(mem_reqs.memoryTypeBits, memory_location)
        if index not in self.allocators:
            print(f'[INFO] Creating allocator for {index}')
            self.allocators[index] = VulkanAllocator(self.vk_device, index, self.mem_properties.memoryTypes[index].propertyFlags)
        mem = self.allocators[index].allocate(mem_reqs.size, mem_reqs.alignment)
        mem.debug_name = self.peek_debug_name(mem)
        if __SHOW_ALLOCATE_AND_FREES__:
            print(f'creating memory for {mem.debug_name}')
        return mem

    def destroy(self):
        if self.allocators is None:
            return
        for k, a in self.allocators.items():
            a.destroy()
        self.allocators = None