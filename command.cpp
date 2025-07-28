#include "command.hpp"
#include "macros/unwrap.hpp"
#include "util/cleaner.hpp"
#include "vk.hpp"

auto allocate_command_buffers(VkDevice device, VkCommandPool command_pool, uint32_t count) -> std::optional<std::vector<VkCommandBuffer>> {
    // TODO: should call vkFreeCommandBuffers
    auto command_buffers = std::vector<VkCommandBuffer>(count);
    vk_args(vkAllocateCommandBuffers(device, &info, command_buffers.data()),
            (VkCommandBufferAllocateInfo{
                .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool        = command_pool,
                .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = count,
            }));
    return command_buffers;
}

auto run_oneshot_command(RunCommandInfo run_info, std::function<bool(VkCommandBuffer)> callback) -> bool {
    unwrap(command_buffers, allocate_command_buffers(run_info.device, run_info.command_pool, 1));
    const auto command_buffers_cleaner = Cleaner{[&] { vkFreeCommandBuffers(run_info.device, run_info.command_pool, command_buffers.size(), command_buffers.data()); }};

    vk_args(vkBeginCommandBuffer(command_buffers[0], &info),
            (VkCommandBufferBeginInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            }));

    ensure(callback(command_buffers[0]));

    vkEndCommandBuffer(command_buffers[0]);
    vk_args(vkQueueSubmit(run_info.queue, 1, &info, VK_NULL_HANDLE),
            (VkSubmitInfo{
                .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = uint32_t(command_buffers.size()),
                .pCommandBuffers    = command_buffers.data(),
            }));
    ensure(vkQueueWaitIdle(run_info.queue) == VK_SUCCESS);
    return true;
}
