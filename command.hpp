#pragma once
#include <functional>
#include <optional>

#include <vulkan/vulkan.h>

struct RunCommandInfo {
    VkDevice      device;
    VkCommandPool command_pool;
    VkQueue       queue;
};

auto allocate_command_buffers(VkDevice device, VkCommandPool command_pool, uint32_t count) -> std::optional<std::vector<VkCommandBuffer>>;
auto run_oneshot_command(RunCommandInfo run_info, std::function<bool(VkCommandBuffer)> callback) -> bool;
