#include <cstring>

#include "buffer.hpp"
#include "image.hpp"
#include "macros/unwrap.hpp"
#include "pixelbuffer.hpp"

auto create_image(VkPhysicalDevice phy, VkDevice device, CreateImageInfo image_info) -> std::optional<CreateImageResult> {
    // create image
    auto image = vk::AutoVkImage();
    vk_args(vkCreateImage(device, &info, nullptr, std::inout_ptr(image)),
            (VkImageCreateInfo{
                .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format    = image_info.format,
                .extent    = {
                       .width  = image_info.width,
                       .height = image_info.height,
                       .depth  = 1,
                },
                .mipLevels     = 1,
                .arrayLayers   = 1,
                .samples       = VK_SAMPLE_COUNT_1_BIT,
                .tiling        = image_info.tiling,
                .usage         = image_info.usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            }));

    // allocate memory
    auto requirements = VkMemoryRequirements();
    vkGetImageMemoryRequirements(device, image.get(), &requirements);

    unwrap(memory_type, find_memory_type(phy, requirements.memoryTypeBits, image_info.props));
    auto memory = vk::AutoVkDeviceMemory();
    vk_args(vkAllocateMemory(device, &info, nullptr, std::inout_ptr(memory)),
            (VkMemoryAllocateInfo{
                .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize  = requirements.size,
                .memoryTypeIndex = memory_type,
            }));

    // bind them
    ensure(vkBindImageMemory(device, image.get(), memory.get(), 0) == VK_SUCCESS);

    return CreateImageResult{std::move(image), std::move(memory)};
}

auto load_image(VkPhysicalDevice phy, VkDevice device, const char* file) -> std::optional<LoadImageResult> {
    // load pixels
    unwrap(pixbuf, PixelBuffer::from_file(file));
    const auto image_size = pixbuf.width * pixbuf.height * 4;

    // create staging buffer
    unwrap_mut(staging_buffer, create_buffer(phy, device,
                                             {
                                                 .size  = image_size,
                                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 .props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             }));
    {
        unwrap_mut(mapping, vk::MemoryMapping::map(device, staging_buffer.memory.get(), image_size));
        std::memcpy(mapping.ptr, pixbuf.data.data(), image_size);
    }

    return LoadImageResult{uint32_t(pixbuf.width), uint32_t(pixbuf.height), std::move(staging_buffer.buffer), std::move(staging_buffer.memory)};
}

auto copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, RunCommandInfo run_info) -> bool {
    ensure(run_oneshot_command(run_info, [=](VkCommandBuffer command_buffer) -> bool {
        vk_args_noret(vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &info),
                      (VkBufferImageCopy{
                          .bufferOffset      = 0,
                          .bufferRowLength   = 0,
                          .bufferImageHeight = 0,
                          .imageSubresource  = {
                               .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                               .mipLevel       = 0,
                               .baseArrayLayer = 0,
                               .layerCount     = 1,
                          },
                          .imageOffset = {0, 0, 0},
                          .imageExtent = {width, height, 1},
                      }));
        return true;
    }));
    return true;
}

auto create_image_view(VkDevice device, VkImage image, CreateImageViewInfo create_info) -> VkImageView_T* {
    auto view = VkImageView();
    vk_args(vkCreateImageView(device, &info, nullptr, &view),
            (VkImageViewCreateInfo{
                .sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image      = image,
                .viewType   = VK_IMAGE_VIEW_TYPE_2D,
                .format     = create_info.format,
                .components = {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = {
                    .aspectMask     = create_info.aspect_flags,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            }));
    return view;
}

