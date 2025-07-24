#pragma once
#include <memory>
#include <optional>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "macros/assert.hpp"

namespace vk {
inline auto default_instance = VkInstance();
inline auto default_device   = VkDevice();

#define declare_autoptr(Name, Type, func)                 \
    struct Name##Deleter {                                \
        static auto operator()(Type* const ptr) -> void { \
            func;                                         \
        }                                                 \
    };                                                    \
    using Auto##Name = std::unique_ptr<Type, Name##Deleter>;

declare_autoptr(VkInstance, VkInstance_T, vkDestroyInstance(ptr, nullptr));
declare_autoptr(VkDevice, VkDevice_T, vkDestroyDevice(ptr, nullptr));
declare_autoptr(VkSurface, VkSurfaceKHR_T, vkDestroySurfaceKHR(default_instance, ptr, nullptr));
declare_autoptr(VkSwapchain, VkSwapchainKHR_T, vkDestroySwapchainKHR(default_device, ptr, nullptr));
declare_autoptr(VkImageView, VkImageView_T, vkDestroyImageView(default_device, ptr, nullptr));
declare_autoptr(VkShaderModule, VkShaderModule_T, vkDestroyShaderModule(default_device, ptr, nullptr));
declare_autoptr(VkPipelineLayout, VkPipelineLayout_T, vkDestroyPipelineLayout(default_device, ptr, nullptr));
declare_autoptr(VkRenderPass, VkRenderPass_T, vkDestroyRenderPass(default_device, ptr, nullptr));
declare_autoptr(VkPipeline, VkPipeline_T, vkDestroyPipeline(default_device, ptr, nullptr));
declare_autoptr(VkFramebuffer, VkFramebuffer_T, vkDestroyFramebuffer(default_device, ptr, nullptr));
declare_autoptr(VkCommandPool, VkCommandPool_T, vkDestroyCommandPool(default_device, ptr, nullptr));
declare_autoptr(VkSemaphore, VkSemaphore_T, vkDestroySemaphore(default_device, ptr, nullptr));
declare_autoptr(VkFence, VkFence_T, vkDestroyFence(default_device, ptr, nullptr));
declare_autoptr(VkBuffer, VkBuffer_T, vkDestroyBuffer(default_device, ptr, nullptr));
declare_autoptr(VkDeviceMemory, VkDeviceMemory_T, vkFreeMemory(default_device, ptr, nullptr));
declare_autoptr(VkDescriptorSetLayout, VkDescriptorSetLayout_T, vkDestroyDescriptorSetLayout(default_device, ptr, nullptr));
declare_autoptr(VkDescriptorPool, VkDescriptorPool_T, vkDestroyDescriptorPool(default_device, ptr, nullptr));

#define vk_args(func, s)            \
    {                               \
        const auto info = s;        \
        ensure(func == VK_SUCCESS); \
    }

#define vk_args_noret(func, s) \
    {                          \
        const auto info = s;   \
        func;                  \
    }

struct MemoryMapping {
    VkDevice       device;
    VkDeviceMemory memory;
    void*          ptr = nullptr;

    static auto map(VkDevice device, VkDeviceMemory memory, size_t size) -> std::optional<MemoryMapping>;

    auto operator=(MemoryMapping&& other) -> MemoryMapping&;

    MemoryMapping() = default;
    MemoryMapping(MemoryMapping&& other);
    ~MemoryMapping();
};

template <class T>
auto query_array(auto func) -> std::optional<std::vector<T>> {
    auto count = uint32_t(0);
    ensure(func(&count, nullptr) == VK_SUCCESS);
    auto arr = std::vector<T>(count);
    if(count > 0) {
        ensure(func(&count, arr.data()) == VK_SUCCESS);
    }
    return arr;
}

auto has_ext(std::span<const VkExtensionProperties> exts, std::string_view req) -> bool;
auto create_shader_module(VkDevice device, const char* spv_file) -> VkShaderModule_T*;

struct SwapchainDetail {
    VkSurfaceCapabilitiesKHR        caps;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   modes;

    static auto query(VkPhysicalDevice device, VkSurfaceKHR surface) -> std::optional<SwapchainDetail>;
};
} // namespace vk
