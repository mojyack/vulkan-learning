#pragma once
#include "vk.hpp"

auto find_memory_type(VkPhysicalDevice phy, uint32_t type_filter, VkMemoryPropertyFlags properties) -> std::optional<uint32_t>;

struct CreateBufferInfo {
    VkDeviceSize          size;
    VkBufferUsageFlags    usage;
    VkMemoryPropertyFlags props;
};

struct CreateBufferResult {
    vk::AutoVkBuffer       buffer;
    vk::AutoVkDeviceMemory memory;
};

auto create_buffer(VkPhysicalDevice phy, VkDevice device, CreateBufferInfo create_info) -> std::optional<CreateBufferResult>;

struct UniformBuffer {
    vk::AutoVkBuffer       buffer;
    vk::AutoVkDeviceMemory memory;
    vk::MemoryMapping      mapping;
};

auto create_uniform_buffers(VkPhysicalDevice phy, VkDevice device, VkDeviceSize size, size_t count) -> std::optional<std::vector<UniformBuffer>>;

struct CopyBufferInfo {
    VkCommandPool command_pool;
    VkQueue       queue;
    VkBuffer      src;
    VkBuffer      dst;
    VkDeviceSize  size;
};

auto copy_buffer(VkDevice device, CopyBufferInfo copy_info) -> bool;

struct TransferMemoryInfo {
    VkPhysicalDevice   phy;
    VkDevice           device;
    VkCommandPool      command_pool;
    VkQueue            queue;
    VkBufferUsageFlags usage;
    const void*        ptr;
    size_t             size;
};

auto transfer_memory(TransferMemoryInfo x_info) -> std::optional<CreateBufferResult>;
