#include "vk.hpp"
#include "macros/unwrap.hpp"
#include "util/file-io.hpp"

namespace vk {
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

auto create_shader_module(const VkDevice device, const char* const spv_file) -> VkShaderModule {
    constexpr auto error_value = nullptr;
    unwrap_v(code, read_file(spv_file));
    const auto create_info = VkShaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode    = std::bit_cast<uint32_t*>(code.data()),
    };
    auto shader_module = VkShaderModule();
    ensure_v(vkCreateShaderModule(device, &create_info, nullptr, &shader_module) == VK_SUCCESS);
    return shader_module;
}
} // namespace vk
