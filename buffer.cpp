#include <cstring>

#include "buffer.hpp"
#include "command.hpp"
#include "macros/unwrap.hpp"

auto find_memory_type(VkPhysicalDevice phy, uint32_t type_filter, VkMemoryPropertyFlags properties) -> std::optional<uint32_t> {
    auto memory_properties = VkPhysicalDeviceMemoryProperties();
    vkGetPhysicalDeviceMemoryProperties(phy, &memory_properties);
    for(auto i = 0u; i < memory_properties.memoryTypeCount; i += 1) {
        if(!(type_filter & (1 << i))) {
            continue;
        }
        if((memory_properties.memoryTypes[i].propertyFlags & properties) != properties) {
            continue;
        }
        return i;
    }
    bail("failed to find suitable memory type");
}

auto create_buffer(VkPhysicalDevice phy, VkDevice device, CreateBufferInfo create_info) -> std::optional<CreateBufferResult> {
    // create buffer
    auto buffer = vk::AutoVkBuffer();
    vk_args(vkCreateBuffer(device, &info, nullptr, std::inout_ptr(buffer)),
            (VkBufferCreateInfo{
                .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size        = create_info.size,
                .usage       = create_info.usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            }));

    // allocate memory
    auto requirements = VkMemoryRequirements();
    vkGetBufferMemoryRequirements(device, buffer.get(), &requirements);

    unwrap(memory_type, find_memory_type(phy, requirements.memoryTypeBits, create_info.props));
    auto memory = vk::AutoVkDeviceMemory();
    vk_args(vkAllocateMemory(device, &info, nullptr, std::inout_ptr(memory)),
            (VkMemoryAllocateInfo{
                .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize  = requirements.size,
                .memoryTypeIndex = memory_type,
            }));

    // bind them
    ensure(vkBindBufferMemory(device, buffer.get(), memory.get(), 0) == VK_SUCCESS);

    return CreateBufferResult{std::move(buffer), std::move(memory)};
}

auto create_uniform_buffers(VkPhysicalDevice phy, VkDevice device, VkDeviceSize size, size_t count) -> std::optional<std::vector<UniformBuffer>> {
    auto ret = std::vector<UniformBuffer>(count);
    for(auto& ubuf : ret) {
        unwrap_mut(buf, create_buffer(phy, device,
                                      {
                                          .size  = size,
                                          .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                          .props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      }));
        unwrap_mut(map, vk::MemoryMapping::map(device, buf.memory.get(), size));
        ubuf = UniformBuffer{std::move(buf.buffer), std::move(buf.memory), std::move(map)};
    }
    return ret;
}

auto copy_buffer(VkDevice device, CopyBufferInfo copy_info) -> bool {
    ensure(run_oneshot_command({device, copy_info.command_pool, copy_info.queue}, [&copy_info](VkCommandBuffer command_buffer) -> bool {
        vk_args_noret(vkCmdCopyBuffer(command_buffer, copy_info.src, copy_info.dst, 1, &info),
                      (VkBufferCopy{
                          .size = copy_info.size,
                      }));
        return true;
    }));
    return true;
}

auto transfer_memory(TransferMemoryInfo x_info) -> std::optional<CreateBufferResult> {
    // create staging buffer
    unwrap_mut(staging_buffer, create_buffer(x_info.phy, x_info.device,
                                             {
                                                 .size  = x_info.size,
                                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 .props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             }));
    // create device local buffer
    unwrap_mut(local_buffer, create_buffer(x_info.phy, x_info.device,
                                           {
                                               .size  = x_info.size,
                                               .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | x_info.usage,
                                               .props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                           }));
    // upload to staging buffer
    {
        unwrap_mut(mapping, vk::MemoryMapping::map(x_info.device, staging_buffer.memory.get(), x_info.size));
        std::memcpy(mapping.ptr, x_info.ptr, x_info.size);
    }
    // copy staging to local buffer
    ensure(copy_buffer(x_info.device,
                       {.command_pool = x_info.command_pool,
                        .queue        = x_info.queue,
                        .src          = staging_buffer.buffer.get(),
                        .dst          = local_buffer.buffer.get(),
                        .size         = x_info.size}));
    return std::move(local_buffer);
}
