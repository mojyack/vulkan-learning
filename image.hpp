#pragma once
#include "command.hpp"
#include "vk.hpp"

struct CreateImageInfo {
    uint32_t              width;
    uint32_t              height;
    VkFormat              format = VK_FORMAT_R8G8B8A8_SRGB;
    VkImageTiling         tiling = VK_IMAGE_TILING_OPTIMAL;
    VkImageUsageFlags     usage  = VK_IMAGE_USAGE_SAMPLED_BIT;
    VkMemoryPropertyFlags props  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
};

struct CreateImageResult {
    vk::AutoVkImage        image;
    vk::AutoVkDeviceMemory memory;
};

auto create_image(VkPhysicalDevice phy, VkDevice device, CreateImageInfo image_info) -> std::optional<CreateImageResult>;

struct LoadImageResult {
    uint32_t               width;
    uint32_t               height;
    vk::AutoVkBuffer       buffer;
    vk::AutoVkDeviceMemory memory;
};

auto load_image(VkPhysicalDevice phy, VkDevice device, const char* file) -> std::optional<LoadImageResult>;

auto copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, RunCommandInfo run_info) -> bool;

struct CreateImageViewInfo {
    VkFormat           format       = VK_FORMAT_R8G8B8A8_SRGB;
    VkImageAspectFlags aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
};

auto create_image_view(VkDevice device, VkImage image, CreateImageViewInfo create_info) -> VkImageView_T*;
