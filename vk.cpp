#include "vk.hpp"
#include "macros/unwrap.hpp"
#include "util/file-io.hpp"

namespace vk {
auto MemoryMapping::map(VkDevice device, VkDeviceMemory memory, size_t size) -> std::optional<MemoryMapping> {
    auto ret   = MemoryMapping();
    ret.device = device;
    ret.memory = memory;
    ensure(vkMapMemory(device, memory, 0, size, 0, &ret.ptr) == VK_SUCCESS);
    return ret;
}

MemoryMapping::MemoryMapping(MemoryMapping&& other) {
    std::swap(device, other.device);
    std::swap(memory, other.memory);
    std::swap(ptr, other.ptr);
}

MemoryMapping::~MemoryMapping() {
    if(ptr != nullptr) {
        vkUnmapMemory(device, memory);
    }
}

auto has_ext(std::span<const VkExtensionProperties> exts, std::string_view req) -> bool {
    for(const auto ext : exts) {
        if(ext.extensionName == req) {
            return true;
        }
    }
    return false;
}

auto SwapchainDetail::query(const VkPhysicalDevice device, const VkSurfaceKHR surface) -> std::optional<SwapchainDetail> {
    auto ret = SwapchainDetail();
    ensure(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &ret.caps) == VK_SUCCESS);
    unwrap_mut(formats, vk::query_array<VkSurfaceFormatKHR>([=](auto... args) { return vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, args...); }));
    ret.formats = std::move(formats);
    unwrap_mut(modes, vk::query_array<VkPresentModeKHR>([=](auto... args) { return vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, args...); }));
    ret.modes = std::move(modes);
    return ret;
}

auto create_shader_module(const VkDevice device, const char* const spv_file) -> VkShaderModule_T* {
    unwrap(code, read_file(spv_file));
    auto shader_module = VkShaderModule();
    vk_args(vkCreateShaderModule(device, &info, nullptr, &shader_module),
            (VkShaderModuleCreateInfo{
                .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = code.size(),
                .pCode    = std::bit_cast<uint32_t*>(code.data()),
            }));
    return shader_module;
}
} // namespace vk
