#include <cstring>
#include <ranges>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "macros/unwrap.hpp"
#include "pixelbuffer.hpp"
#include "util/cleaner.hpp"
#include "vk.hpp"

namespace {
struct Vertex {
    glm::vec2 pos;
    glm::vec2 tex_coord;
    glm::vec3 color;
};

const auto vertex_binding_desc = VkVertexInputBindingDescription{
    .binding   = 0,
    .stride    = sizeof(Vertex),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
};

const auto vertex_attr_desc = std::array{
    VkVertexInputAttributeDescription{
        .location = 0,
        .binding  = 0,
        .format   = VK_FORMAT_R32G32_SFLOAT,
        .offset   = offsetof(Vertex, pos),
    },
    VkVertexInputAttributeDescription{
        .location = 1,
        .binding  = 0,
        .format   = VK_FORMAT_R32G32B32_SFLOAT,
        .offset   = offsetof(Vertex, color),
    },
    VkVertexInputAttributeDescription{
        .location = 2,
        .binding  = 0,
        .format   = VK_FORMAT_R32G32_SFLOAT,
        .offset   = offsetof(Vertex, tex_coord),
    },
};

const auto vertices = std::array{
    Vertex{{-0.5, -0.5}, {1, 0}, {1, 0, 0}},
    Vertex{{0.5, -0.5}, {0, 0}, {0, 1, 0}},
    Vertex{{0.5, 0.5}, {0, 1}, {0, 0, 1}},
    Vertex{{-0.5, 0.5}, {1, 1}, {1, 1, 1}},
};

const auto indices = std::array<uint16_t, 6>{0, 1, 2, 2, 3, 0};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

constexpr auto invalid_queue_index = (uint32_t)-1;

const auto required_exts = std::array{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

auto create_instance() -> VkInstance_T* {
    auto app_info = VkApplicationInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = "vk triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName        = "None",
        .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion         = VK_API_VERSION_1_0,
    };

    auto glfw_exts_count = uint32_t(0);
    auto glfw_exts       = glfwGetRequiredInstanceExtensions(&glfw_exts_count);
    ensure(glfw_exts);
    PRINT("glfw exts:");
    for(auto i = 0u; i < glfw_exts_count; i += 1) {
        std::println("- {}", glfw_exts[i]);
    }

    auto instance_create_info = VkInstanceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo        = &app_info,
        .enabledExtensionCount   = glfw_exts_count,
        .ppEnabledExtensionNames = glfw_exts,
    };
    if(true) {
        static const auto layers = std::array{"VK_LAYER_KHRONOS_validation"};

        instance_create_info.ppEnabledLayerNames = layers.data();
        instance_create_info.enabledLayerCount   = layers.size();
    } else {
        instance_create_info.enabledLayerCount = 0;
    }

    ensure(vkCreateInstance(&instance_create_info, nullptr, &vk::default_instance) == VK_SUCCESS);
    return vk::default_instance;
}

auto pickup_phy(VkInstance instance, VkSurfaceKHR surface) -> VkPhysicalDevice_T* {
    unwrap(devices, vk::query_array<VkPhysicalDevice>([instance](auto... args) { return vkEnumeratePhysicalDevices(instance, args...); }));
    ensure(!devices.empty());

    for(const auto dev : devices) {
        auto props = VkPhysicalDeviceProperties();
        auto feats = VkPhysicalDeviceFeatures();
        vkGetPhysicalDeviceProperties(dev, &props);
        vkGetPhysicalDeviceFeatures(dev, &feats);
        unwrap(exts, vk::query_array<VkExtensionProperties>([dev](auto... args) { return vkEnumerateDeviceExtensionProperties(dev, nullptr, args...); }));
        PRINT("dev: name={} version={} exts={}", props.deviceName, props.apiVersion, exts.size());
        if(props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            continue;
        }
        if(!feats.geometryShader) {
            continue;
        }
        auto exts_complete = true;
        for(const auto req : required_exts) {
            exts_complete &= vk::has_ext(exts, req);
        }
        if(!exts_complete) {
            continue;
        }
        unwrap(sc, vk::SwapchainDetail::query(dev, surface));
        if(sc.formats.empty() || sc.modes.empty()) {
            continue;
        }
        return dev;
    }
    return VK_NULL_HANDLE;
}

auto pickup_queues(VkPhysicalDevice phy, VkSurfaceKHR surface) -> std::optional<std::array<uint32_t, 2>> {
    // TODO: this is an experimental implementation

    unwrap(queue_families, vk::query_array<VkQueueFamilyProperties>([phy](auto... args) {vkGetPhysicalDeviceQueueFamilyProperties(phy, args...); return VK_SUCCESS; }));
    PRINT("queues={}", queue_families.size());
    auto graphics_queue_index = invalid_queue_index;
    auto present_queue_index  = invalid_queue_index;
    for(auto i = 0uz; i < queue_families.size(); i += 1) {
        const auto& family = queue_families[i];

        auto support_present = VkBool32(false);
        ensure(vkGetPhysicalDeviceSurfaceSupportKHR(phy, i, surface, &support_present) == VK_SUCCESS);

        std::println("queue {}: flags={:02x} graphics?={} present?={}", i, family.queueFlags, family.queueFlags & VK_QUEUE_GRAPHICS_BIT, support_present);
        if(family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphics_queue_index = i;
        }
        if(support_present) {
            present_queue_index = i;
        }
    }
    return std::array{graphics_queue_index, present_queue_index};
}

auto create_device(VkPhysicalDevice phy, std::span<const uint32_t> queue_indices) -> VkDevice_T* {
    auto queue_create_infos = std::vector<VkDeviceQueueCreateInfo>();
    for(const auto i : queue_indices) {
        static const auto queue_priority = 1.0f;
        queue_create_infos.emplace_back(VkDeviceQueueCreateInfo{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = uint32_t(i),
            .queueCount       = 1,
            .pQueuePriorities = &queue_priority,
        });
    }
    const auto device_features = VkPhysicalDeviceFeatures{};
    vk_args(vkCreateDevice(phy, &info, nullptr, &vk::default_device),
            (VkDeviceCreateInfo{
                .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .queueCreateInfoCount    = uint32_t(queue_create_infos.size()),
                .pQueueCreateInfos       = queue_create_infos.data(),
                .enabledExtensionCount   = required_exts.size(),
                .ppEnabledExtensionNames = required_exts.data(),
                .pEnabledFeatures        = &device_features,
            }));
    PRINT("logical device created");
    return vk::default_device;
}

struct SwapchainParams {
    VkSurfaceFormatKHR       format;
    VkSurfaceCapabilitiesKHR caps;
    VkExtent2D               extent;
};

auto find_optimal_swapchain_params(GLFWwindow& window, VkPhysicalDevice phy, VkSurfaceKHR surface) -> std::optional<SwapchainParams> {
    unwrap(support, vk::SwapchainDetail::query(phy, surface));
    auto params = SwapchainParams{
        .format = support.formats[0],
        .caps   = support.caps,
        .extent = support.caps.currentExtent,
    };
    for(const auto& format : support.formats) {
        if(format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            params.format = format;
            break;
        }
    }
    if(params.extent.width == std::numeric_limits<uint32_t>::max()) {
        auto fb = std::array<int, 2>();
        glfwGetFramebufferSize(&window, &fb[0], &fb[1]);
        const auto& caps     = support.caps;
        params.extent.width  = std::clamp<uint32_t>(fb[0], caps.minImageExtent.width, caps.maxImageExtent.width);
        params.extent.height = std::clamp<uint32_t>(fb[1], caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    return params;
}

auto create_swapchain(VkDevice device, VkSurfaceKHR surface, std::span<const uint32_t> queue_indices, const SwapchainParams& swapchain_params) -> VkSwapchainKHR_T* {
    auto swapchain_image_count = swapchain_params.caps.minImageCount + 1;
    if(swapchain_params.caps.maxImageCount != 0 && swapchain_image_count > swapchain_params.caps.maxImageCount) {
        swapchain_image_count = swapchain_params.caps.maxImageCount;
    }

    auto swapchain_create_info = VkSwapchainCreateInfoKHR{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = surface,
        .minImageCount    = swapchain_image_count,
        .imageFormat      = swapchain_params.format.format,
        .imageColorSpace  = swapchain_params.format.colorSpace,
        .imageExtent      = swapchain_params.extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = swapchain_params.caps.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = VK_PRESENT_MODE_FIFO_KHR,
        .clipped          = VK_TRUE,
    };
    auto all_same = true;
    for(auto i = 1uz; i < queue_indices.size(); i += 1) {
        if(queue_indices[0] != queue_indices[i]) {
            all_same = false;
            break;
        }
    }
    if(all_same) {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    } else {
        swapchain_create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        swapchain_create_info.queueFamilyIndexCount = queue_indices.size();
        swapchain_create_info.pQueueFamilyIndices   = queue_indices.data();
    }

    auto swapchain = VkSwapchainKHR();
    ensure(vkCreateSwapchainKHR(device, &swapchain_create_info, nullptr, &swapchain) == VK_SUCCESS);
    PRINT("swapchain created");
    return swapchain;
}

auto create_image_view(VkDevice device, VkImage image, VkFormat format) -> VkImageView_T* {
    auto view = VkImageView();
    vk_args(vkCreateImageView(device, &info, nullptr, &view),
            (VkImageViewCreateInfo{
                .sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image      = image,
                .viewType   = VK_IMAGE_VIEW_TYPE_2D,
                .format     = format,
                .components = {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            }));
    return view;
}

auto create_image_views(VkDevice device, VkSwapchainKHR swapchain, const VkFormat& swapchain_format) -> std::optional<std::vector<vk::AutoVkImageView>> {
    // retrieve images from swapchain
    unwrap(swapchain_images, vk::query_array<VkImage>([&](auto... args) { return vkGetSwapchainImagesKHR(device, swapchain, args...); }));
    PRINT("images={}", swapchain_images.size());

    // create image views
    auto image_views = std::vector<vk::AutoVkImageView>(swapchain_images.size());
    for(auto&& [image, view] : std::ranges::zip_view(swapchain_images, image_views)) {
        unwrap_mut(v, create_image_view(device, image, swapchain_format));
        view.reset(&v);
    }
    return image_views;
}

auto create_texture_sampler(VkDevice device) -> VkSampler_T* {
    auto ret = VkSampler();
    vk_args(vkCreateSampler(device, &info, nullptr, &ret),
            (VkSamplerCreateInfo{
                .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter               = VK_FILTER_LINEAR,
                .minFilter               = VK_FILTER_LINEAR,
                .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .mipLodBias              = 0.0f,
                .anisotropyEnable        = VK_FALSE,
                .maxAnisotropy           = 1.0f,
                .compareEnable           = VK_FALSE,
                .compareOp               = VK_COMPARE_OP_ALWAYS,
                .minLod                  = 0.0f,
                .maxLod                  = 0.0f,
                .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                .unnormalizedCoordinates = VK_FALSE,
            }));
    return ret;
}

auto create_desc_set_layout(VkDevice device) -> VkDescriptorSetLayout_T* {
    static const auto bindings = std::array{
        VkDescriptorSetLayoutBinding{
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT,
        },
        VkDescriptorSetLayoutBinding{
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };
    auto ret = VkDescriptorSetLayout();
    vk_args(vkCreateDescriptorSetLayout(device, &info, nullptr, &ret),
            (VkDescriptorSetLayoutCreateInfo{
                .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = uint32_t(bindings.size()),
                .pBindings    = bindings.data(),
            }));
    return ret;
}

auto create_descriptror_pool(VkDevice device, uint32_t count) -> VkDescriptorPool_T* {
    const auto pool_sizes = std::array{
        VkDescriptorPoolSize{
            .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = count,
        },
        VkDescriptorPoolSize{
            .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = count,
        },
    };
    auto ret = VkDescriptorPool();
    vk_args(vkCreateDescriptorPool(device, &info, nullptr, &ret),
            (VkDescriptorPoolCreateInfo{
                .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets       = count,
                .poolSizeCount = uint32_t(pool_sizes.size()),
                .pPoolSizes    = pool_sizes.data(),
            }));
    return ret;
};

auto allocate_descriptor_sets(VkDevice device, VkDescriptorPool desc_pool, VkDescriptorSetLayout desc_set_layout, uint32_t count) -> std::optional<std::vector<VkDescriptorSet>> {
    const auto layouts = std::vector(count, desc_set_layout);
    auto       ret     = std::vector<VkDescriptorSet>(count);
    vk_args(vkAllocateDescriptorSets(device, &info, ret.data()),
            (VkDescriptorSetAllocateInfo{
                .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool     = desc_pool,
                .descriptorSetCount = count,
                .pSetLayouts        = layouts.data(),
            }));
    return ret;
}

auto create_pipeline_layout(VkDevice device, VkDescriptorSetLayout desc_set_layout) -> VkPipelineLayout_T* {
    auto pipeline_layout = VkPipelineLayout();
    vk_args(vkCreatePipelineLayout(device, &info, nullptr, &pipeline_layout),
            (VkPipelineLayoutCreateInfo{
                .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts    = &desc_set_layout,
            }));
    return pipeline_layout;
}

auto create_render_pass(VkDevice device, VkFormat format) -> VkRenderPass_T* {
    // attachments
    const auto color_attachment = VkAttachmentDescription{
        .format         = format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    const auto color_attachment_ref = VkAttachmentReference{
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    const auto subpass = VkSubpassDescription{
        .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &color_attachment_ref,
    };

    // render pass
    const auto subpass_dep = VkSubpassDependency{
        .srcSubpass    = VK_SUBPASS_EXTERNAL,
        .dstSubpass    = 0,
        .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    auto render_pass = VkRenderPass();
    vk_args(vkCreateRenderPass(device, &info, nullptr, &render_pass),
            (VkRenderPassCreateInfo{
                .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments    = &color_attachment,
                .subpassCount    = 1,
                .pSubpasses      = &subpass,
                .dependencyCount = 1,
                .pDependencies   = &subpass_dep,
            }));
    return render_pass;
}

auto create_pipeline(VkDevice device, VkRenderPass render_pass, VkPipelineLayout pipeline_layout, VkExtent2D swapchain_extent, VkShaderModule vertex_shader, VkShaderModule fragment_shader) -> VkPipeline_T* {
    // input assembly
    const auto input_assembly_create_info = VkPipelineInputAssemblyStateCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    // viewport
    const auto viewport = VkViewport{
        .x        = 0,
        .y        = 0,
        .width    = float(swapchain_extent.width),
        .height   = float(swapchain_extent.height),
        .minDepth = 0,
        .maxDepth = 1,
    };
    const auto scissor = VkRect2D{
        .offset = {0, 0},
        .extent = swapchain_extent,
    };
    // dynamic state
    const auto dynamic_states            = std::array{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const auto dynamic_state_create_info = VkPipelineDynamicStateCreateInfo{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = uint32_t(dynamic_states.size()),
        .pDynamicStates    = dynamic_states.data(),
    };
    const auto viewport_state_create_info = VkPipelineViewportStateCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        // .pViewports = &viewport,
        .scissorCount = 1,
        // .pScissors = &scissor,
    };
    // rasterizer
    const auto rasterizer_state_create_info = VkPipelineRasterizationStateCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable        = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode             = VK_POLYGON_MODE_FILL,
        .cullMode                = VK_CULL_MODE_BACK_BIT,
        .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable         = VK_FALSE,
        .lineWidth               = 1.0,
    };
    // multisample
    const auto multisample_state_create_info = VkPipelineMultisampleStateCreateInfo{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable  = VK_FALSE,
    };
    // color blending
    const auto color_blend_attach_state = VkPipelineColorBlendAttachmentState{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const auto color_blend_state_create_info = VkPipelineColorBlendStateCreateInfo{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable   = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments    = &color_blend_attach_state,
    };

    // pipeline
    const auto vertex_input_state_create_info = VkPipelineVertexInputStateCreateInfo{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertex_binding_desc,
        .vertexAttributeDescriptionCount = uint32_t(vertex_attr_desc.size()),
        .pVertexAttributeDescriptions    = vertex_attr_desc.data(),
    };
    const auto shader_stages = std::array<VkPipelineShaderStageCreateInfo, 2>{{
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader,
            .pName  = "main",
        },
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader,
            .pName  = "main",
        },
    }};

    auto pipeline = VkPipeline();
    vk_args(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline),
            (VkGraphicsPipelineCreateInfo{
                .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount          = uint32_t(shader_stages.size()),
                .pStages             = shader_stages.data(),
                .pVertexInputState   = &vertex_input_state_create_info,
                .pInputAssemblyState = &input_assembly_create_info,
                .pViewportState      = &viewport_state_create_info,
                .pRasterizationState = &rasterizer_state_create_info,
                .pMultisampleState   = &multisample_state_create_info,
                .pDepthStencilState  = nullptr,
                .pColorBlendState    = &color_blend_state_create_info,
                .pDynamicState       = &dynamic_state_create_info,
                .layout              = pipeline_layout,
                .renderPass          = render_pass,
                .subpass             = 0,
            }));
    return pipeline;
}

auto create_framebuffers(VkDevice device, std::span<const vk::AutoVkImageView> image_views, VkRenderPass render_pass, VkExtent2D swapchain_extent) -> std::optional<std::vector<vk::AutoVkFramebuffer>> {
    auto framebuffers = std::vector<vk::AutoVkFramebuffer>(image_views.size());
    for(auto&& [fb, image] : std::ranges::zip_view(framebuffers, image_views)) {
        const auto attachments = std::array{image.get()};
        vk_args(vkCreateFramebuffer(device, &info, nullptr, std::inout_ptr(fb)),
                (VkFramebufferCreateInfo{
                    .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass      = render_pass,
                    .attachmentCount = uint32_t(attachments.size()),
                    .pAttachments    = attachments.data(),
                    .width           = swapchain_extent.width,
                    .height          = swapchain_extent.height,
                    .layers          = 1,
                }));
    }
    return framebuffers;
}

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

struct CreateBufferInfo {
    VkDeviceSize          size;
    VkBufferUsageFlags    usage;
    VkMemoryPropertyFlags props;
};

struct CreateBufferResult {
    vk::AutoVkBuffer       buffer;
    vk::AutoVkDeviceMemory memory;
};

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

struct UniformBuffer {
    vk::AutoVkBuffer       buffer;
    vk::AutoVkDeviceMemory memory;
    vk::MemoryMapping      mapping;
};

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

struct CreateImageInfo {
    uint32_t              width;
    uint32_t              height;
    VkImageTiling         tiling = VK_IMAGE_TILING_OPTIMAL;
    VkImageUsageFlags     usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkMemoryPropertyFlags props  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
};

struct CreateImageResult {
    vk::AutoVkImage        image;
    vk::AutoVkDeviceMemory memory;
};

auto create_image(VkPhysicalDevice phy, VkDevice device, CreateImageInfo image_info) -> std::optional<CreateImageResult> {
    // create image
    auto image = vk::AutoVkImage();
    vk_args(vkCreateImage(device, &info, nullptr, std::inout_ptr(image)),
            (VkImageCreateInfo{
                .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format    = VK_FORMAT_R8G8B8A8_SRGB,
                .extent    = {
                       .width  = image_info.width,
                       .height = image_info.height,
                       .depth  = 1,
                },
                .mipLevels     = 1,
                .arrayLayers   = 1,
                .samples       = VK_SAMPLE_COUNT_1_BIT,
                .tiling        = image_info.tiling,
                .usage         = image_info.usage,
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

struct LoadImageResult {
    uint32_t               width;
    uint32_t               height;
    vk::AutoVkBuffer       buffer;
    vk::AutoVkDeviceMemory memory;
};

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

struct RunCommandInfo {
    VkDevice      device;
    VkCommandPool command_pool;
    VkQueue       queue;
};

auto run_oneshot_command(RunCommandInfo run_info, auto callback) -> bool {
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

auto transition_image_layout(VkImage image, VkFormat format, VkImageLayout from, VkImageLayout to, RunCommandInfo run_info) -> bool {
    struct Entry {
        VkImageLayout        from;
        VkImageLayout        to;
        VkPipelineStageFlags src_stage;
        VkPipelineStageFlags dst_stage;
        VkAccessFlags        src_access;
        VkAccessFlags        dst_access;
    };
    static const auto table = std::array{
        Entry{
            .from       = VK_IMAGE_LAYOUT_UNDEFINED,
            .to         = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .src_stage  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dst_stage  = VK_PIPELINE_STAGE_TRANSFER_BIT,
            .src_access = 0,
            .dst_access = VK_ACCESS_TRANSFER_WRITE_BIT,
        },
        Entry{
            .from       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .to         = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .src_stage  = VK_PIPELINE_STAGE_TRANSFER_BIT,
            .dst_stage  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .src_access = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dst_access = VK_ACCESS_SHADER_READ_BIT,
        },
    };

    auto conf = (const Entry*)(nullptr);
    for(const auto& entry : table) {
        if(entry.from == from && entry.to == to) {
            conf = &entry;
            break;
        }
    }
    ensure(conf);

    ensure(run_oneshot_command(run_info, [=](VkCommandBuffer command_buffer) {
        vk_args_noret(vkCmdPipelineBarrier(command_buffer, conf->src_stage, conf->dst_stage, 0, 0, nullptr, 0, nullptr, 1, &info),
                      (VkImageMemoryBarrier{
                          .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                          .srcAccessMask       = conf->src_access,
                          .dstAccessMask       = conf->dst_access,
                          .oldLayout           = from,
                          .newLayout           = to,
                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .image               = image,
                          .subresourceRange    = {
                                 .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel   = 0,
                                 .levelCount     = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount     = 1,
                          },
                      }));
        return true;
    }));
    return true;
}

auto create_texture_image(VkPhysicalDevice phy, VkDevice device, VkCommandPool command_pool, VkQueue queue, const char* file) -> std::optional<CreateImageResult> {
    unwrap_mut(pix, load_image(phy, device, file));
    unwrap_mut(buffer, create_image(phy, device,
                                    {
                                        .width  = pix.width,
                                        .height = pix.height,
                                    }));
    ensure(transition_image_layout(buffer.image.get(), VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {device, command_pool, queue}));
    ensure(copy_buffer_to_image(pix.buffer.get(), buffer.image.get(), pix.width, pix.height, {device, command_pool, queue}));
    ensure(transition_image_layout(buffer.image.get(), VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {device, command_pool, queue}));
    return std::move(buffer);
}

auto create_command_pool(VkDevice device, uint32_t graphics_queue_index) -> VkCommandPool_T* {
    auto command_pool = VkCommandPool();
    vk_args(vkCreateCommandPool(device, &info, nullptr, &command_pool),
            (VkCommandPoolCreateInfo{
                .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = uint32_t(graphics_queue_index),
            }));
    return command_pool;
}

struct CopyBufferInfo {
    VkCommandPool command_pool;
    VkQueue       queue;
    VkBuffer      src;
    VkBuffer      dst;
    VkDeviceSize  size;
};

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

auto create_semaphores(VkDevice device, uint32_t count) -> std::optional<std::vector<vk::AutoVkSemaphore>> {
    auto ret = std::vector<vk::AutoVkSemaphore>(count);
    for(auto& r : ret) {
        vk_args(vkCreateSemaphore(device, &info, nullptr, std::inout_ptr(r)),
                (VkSemaphoreCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                }));
    }
    return ret;
}

auto create_fences(VkDevice device, uint32_t count, bool signaled) -> std::optional<std::vector<vk::AutoVkFence>> {
    auto ret = std::vector<vk::AutoVkFence>(count);
    for(auto& r : ret) {
        vk_args(vkCreateFence(device, &info, nullptr, std::inout_ptr(r)),
                (VkFenceCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                    .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : VkFenceCreateFlagBits(0),
                }));
    }
    return ret;
}

struct RecordCommandBufferInfo {
    VkPipeline       pipeline;
    VkRenderPass     render_pass;
    VkBuffer         vertex_buffer;
    VkBuffer         index_buffer;
    VkFramebuffer    framebuffer;
    VkExtent2D       extent;
    VkCommandBuffer  command_buffer;
    VkPipelineLayout pipeline_layout;
    VkDescriptorSet  desc_set;
};

auto record_command_buffer(RecordCommandBufferInfo rec_info) -> bool {
    vk_args(vkBeginCommandBuffer(rec_info.command_buffer, &info),
            (VkCommandBufferBeginInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0,
            }));
    const auto clear_color = VkClearValue{.color = {.float32 = {0, 0, 0, 1}}};
    vk_args_noret(vkCmdBeginRenderPass(rec_info.command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE),
                  (VkRenderPassBeginInfo{
                      .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                      .renderPass  = rec_info.render_pass,
                      .framebuffer = rec_info.framebuffer,
                      .renderArea  = {
                           .offset = {0, 0},
                           .extent = rec_info.extent,
                      },
                      .clearValueCount = 1,
                      .pClearValues    = &clear_color,
                  }));
    vkCmdBindPipeline(rec_info.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rec_info.pipeline);
    const auto vertex_buffers = std::array{rec_info.vertex_buffer};
    const auto offsets        = std::array{VkDeviceSize(0)};
    vkCmdBindVertexBuffers(rec_info.command_buffer, 0, 1, vertex_buffers.data(), offsets.data());
    vkCmdBindIndexBuffer(rec_info.command_buffer, rec_info.index_buffer, 0, VK_INDEX_TYPE_UINT16);
    vk_args_noret(vkCmdSetViewport(rec_info.command_buffer, 0, 1, &info),
                  (VkViewport{
                      .x        = 0,
                      .y        = 0,
                      .width    = float(rec_info.extent.width),
                      .height   = float(rec_info.extent.height),
                      .minDepth = 0,
                      .maxDepth = 1,
                  }));
    vk_args_noret(vkCmdSetScissor(rec_info.command_buffer, 0, 1, &info),
                  (VkRect2D{
                      .offset = {0, 0},
                      .extent = rec_info.extent,
                  }));
    vkCmdBindDescriptorSets(rec_info.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rec_info.pipeline_layout, 0, 1, &rec_info.desc_set, 0, nullptr);
    vkCmdDrawIndexed(rec_info.command_buffer, indices.size(), 1, 0, 0, 0);
    vkCmdEndRenderPass(rec_info.command_buffer);
    ensure(vkEndCommandBuffer(rec_info.command_buffer) == VK_SUCCESS);
    return true;
};

struct Context {
    vk::AutoVkInstance                 instance;
    vk::AutoVkSurface                  surface;
    VkPhysicalDevice                   phy;
    uint32_t                           graphics_queue_index;
    uint32_t                           present_queue_index;
    vk::AutoVkDevice                   device;
    SwapchainParams                    swapchain_params;
    vk::AutoVkSwapchain                swapchain;
    std::vector<vk::AutoVkImageView>   swapchain_images;
    vk::AutoVkShaderModule             vertex_shader;
    vk::AutoVkShaderModule             fragment_shader;
    vk::AutoVkDescriptorSetLayout      desc_set_layout;
    vk::AutoVkPipelineLayout           pipeline_layout;
    vk::AutoVkRenderPass               render_pass;
    vk::AutoVkPipeline                 pipeline;
    std::vector<vk::AutoVkFramebuffer> framebuffers;
    vk::AutoVkCommandPool              command_pool;
    std::vector<VkCommandBuffer>       command_buffers;
    vk::AutoVkDescriptorPool           desc_pool;
    std::vector<VkDescriptorSet>       desc_sets;
    std::vector<vk::AutoVkSemaphore>   image_avail_semaphores;
    std::vector<vk::AutoVkSemaphore>   render_finished_semaphores;
    std::vector<vk::AutoVkFence>       in_flight_fences;
    bool                               window_resized = false;
};

auto on_window_resize(GLFWwindow* window, int width, int height) -> void {
    auto& context = *std::bit_cast<Context*>(glfwGetWindowUserPointer(window));
    PRINT("window resized {}x{}", width, height);
    context.window_resized = true;
}

auto recreate_swapchain(GLFWwindow& window, Context& context) -> bool {
    PRINT("recreating swapchain");

    // handle minimization
    {
        auto width  = 0;
        auto height = 0;
        glfwGetFramebufferSize(&window, &width, &height);
        while(width == 0 || height == 0) {
            glfwGetFramebufferSize(&window, &width, &height);
            glfwWaitEvents();
        }
    }

    context.framebuffers.clear();
    context.swapchain_images.clear();
    context.swapchain.reset();

    ensure(vkDeviceWaitIdle(context.device.get()) == VK_SUCCESS);
    const auto queue_indices = std::array{context.graphics_queue_index, context.present_queue_index};
    unwrap(swapchain_params, find_optimal_swapchain_params(window, context.phy, context.surface.get()));
    context.swapchain_params = swapchain_params;
    unwrap_mut(swapchain, create_swapchain(context.device.get(), context.surface.get(), queue_indices, context.swapchain_params));
    context.swapchain.reset(&swapchain);
    unwrap_mut(swapchain_images, create_image_views(context.device.get(), context.swapchain.get(), context.swapchain_params.format.format));
    context.swapchain_images = std::move(swapchain_images);
    unwrap_mut(framebuffers, create_framebuffers(context.device.get(), context.swapchain_images, context.render_pass.get(), context.swapchain_params.extent));
    context.framebuffers = std::move(framebuffers);
    return true;
}

struct TransferMemoryInfo {
    VkPhysicalDevice   phy;
    VkDevice           device;
    VkCommandPool      command_pool;
    VkQueue            queue;
    VkBufferUsageFlags usage;
    const void*        ptr;
    size_t             size;
};

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

auto vulkan_main(GLFWwindow& window) -> bool {
    auto context = Context();

    glfwSetWindowUserPointer(&window, &context);
    glfwSetFramebufferSizeCallback(&window, on_window_resize);

    {
        unwrap_mut(instance, create_instance());
        context.instance.reset(&instance);
    }
    ensure(glfwCreateWindowSurface(context.instance.get(), &window, nullptr, std::inout_ptr(context.surface)) == VK_SUCCESS);
    {
        unwrap_mut(phy, pickup_phy(context.instance.get(), context.surface.get()));
        context.phy = &phy;
    }
    {
        unwrap(indices, pickup_queues(context.phy, context.surface.get()));
        const auto [graphics, present] = indices;
        ensure(graphics >= 0);
        ensure(present >= 0);
        context.graphics_queue_index = graphics;
        context.present_queue_index  = present;
    }
    const auto queue_indices = std::array{context.graphics_queue_index, context.present_queue_index};
    {
        unwrap_mut(device, create_device(context.phy, queue_indices));
        context.device.reset(&device);
    }
    {
        unwrap(swapchain_params, find_optimal_swapchain_params(window, context.phy, context.surface.get()));
        context.swapchain_params = swapchain_params;
    }
    {
        unwrap_mut(swapchain, create_swapchain(context.device.get(), context.surface.get(), queue_indices, context.swapchain_params));
        context.swapchain.reset(&swapchain);
    }
    {
        unwrap_mut(swapchain_images, create_image_views(context.device.get(), context.swapchain.get(), context.swapchain_params.format.format));
        context.swapchain_images = std::move(swapchain_images);
    }
    {
        unwrap_mut(vert, vk::create_shader_module(context.device.get(), "build/vert.spv"));
        context.vertex_shader.reset(&vert);
        unwrap_mut(frag, vk::create_shader_module(context.device.get(), "build/frag.spv"));
        context.fragment_shader.reset(&frag);
    }
    {
        unwrap_mut(render_pass, create_render_pass(context.device.get(), context.swapchain_params.format.format));
        context.render_pass.reset(&render_pass);
    }
    {
        unwrap_mut(desc_set_layout, create_desc_set_layout(context.device.get()));
        context.desc_set_layout.reset(&desc_set_layout);
    }
    {
        unwrap_mut(pipeline_layout, create_pipeline_layout(context.device.get(), context.desc_set_layout.get()));
        context.pipeline_layout.reset(&pipeline_layout);
    }
    {
        unwrap_mut(pipeline, create_pipeline(context.device.get(), context.render_pass.get(), context.pipeline_layout.get(), context.swapchain_params.extent, context.vertex_shader.get(), context.fragment_shader.get()));
        context.pipeline.reset(&pipeline);
    }
    {
        unwrap_mut(framebuffers, create_framebuffers(context.device.get(), context.swapchain_images, context.render_pass.get(), context.swapchain_params.extent));
        context.framebuffers = std::move(framebuffers);
    }
    constexpr auto max_frames_in_flight = 2;
    {
        unwrap_mut(command_pool, create_command_pool(context.device.get(), context.graphics_queue_index));
        context.command_pool.reset(&command_pool);
    }
    {
        unwrap_mut(command_buffers, allocate_command_buffers(context.device.get(), context.command_pool.get(), max_frames_in_flight));
        context.command_buffers = std::move(command_buffers);
    }
    {
        unwrap_mut(desc_pool, create_descriptror_pool(context.device.get(), max_frames_in_flight));
        context.desc_pool.reset(&desc_pool);
    }
    {
        unwrap_mut(desc_sets, allocate_descriptor_sets(context.device.get(), context.desc_pool.get(), context.desc_set_layout.get(), max_frames_in_flight));
        context.desc_sets = std::move(desc_sets);
    }
    {
        unwrap_mut(image_avail_semaphores, create_semaphores(context.device.get(), max_frames_in_flight));
        unwrap_mut(render_finished_semaphores, create_semaphores(context.device.get(), max_frames_in_flight));
        unwrap_mut(in_flight_fences, create_fences(context.device.get(), max_frames_in_flight, true));
        context.image_avail_semaphores     = std::move(image_avail_semaphores);
        context.render_finished_semaphores = std::move(render_finished_semaphores);
        context.in_flight_fences           = std::move(in_flight_fences);
    }

    auto graphics_queue = VkQueue();
    auto present_queue  = VkQueue();
    vkGetDeviceQueue(context.device.get(), context.graphics_queue_index, 0, &graphics_queue);
    vkGetDeviceQueue(context.device.get(), context.present_queue_index, 0, &present_queue);

    // upload vertex buffer
    unwrap(vertex_buffer, transfer_memory({
                              .phy          = context.phy,
                              .device       = context.device.get(),
                              .command_pool = context.command_pool.get(),
                              .queue        = graphics_queue,
                              .usage        = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              .ptr          = vertices.data(),
                              .size         = sizeof(vertices[0]) * vertices.size(),
                          }));
    // upload index buffer
    unwrap(index_buffer, transfer_memory({
                             .phy          = context.phy,
                             .device       = context.device.get(),
                             .command_pool = context.command_pool.get(),
                             .queue        = graphics_queue,
                             .usage        = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             .ptr          = indices.data(),
                             .size         = sizeof(indices[0]) * indices.size(),
                         }));
    // allocate uniform buffer
    unwrap(uniform_buffers, create_uniform_buffers(context.phy, context.device.get(), sizeof(UniformBufferObject), max_frames_in_flight));
    // create texture
    unwrap_mut(tex, create_texture_image(context.phy, context.device.get(), context.command_pool.get(), graphics_queue, "textures/icon.png"));
    unwrap_mut(tex_view, create_image_view(context.device.get(), tex.image.get(), VK_FORMAT_R8G8B8A8_SRGB));
    const auto auto_tex_view = vk::AutoVkImageView(&tex_view);
    unwrap_mut(sampler, create_texture_sampler(context.device.get()));
    const auto auto_sampler = vk::AutoVkSampler(&sampler);
    // update descriptor set
    for(auto&& [set, ubuf] : std::ranges::zip_view(context.desc_sets, uniform_buffers)) {
        const auto buffer_info = VkDescriptorBufferInfo{
            .buffer = ubuf.buffer.get(),
            .offset = 0,
            .range  = sizeof(UniformBufferObject),
        };
        const auto image_info = VkDescriptorImageInfo{
            .sampler     = &sampler,
            .imageView   = &tex_view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const auto desc_writes = std::array{
            VkWriteDescriptorSet{
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = set,
                .dstBinding      = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &buffer_info,
            },
            VkWriteDescriptorSet{
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = set,
                .dstBinding      = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &image_info,
            },
        };
        vkUpdateDescriptorSets(context.device.get(), desc_writes.size(), desc_writes.data(), 0, nullptr);
    }

    auto count = 0uz;

    while(!glfwWindowShouldClose(&window)) {
        glfwPollEvents();

        constexpr auto uint64_max = std::numeric_limits<uint64_t>::max();

        // wait for previous command completion
        const auto current_frame = uint32_t(count % max_frames_in_flight);
        const auto fence         = context.in_flight_fences[current_frame].get();
        ensure(vkWaitForFences(context.device.get(), 1, &fence, VK_TRUE, uint64_max) == VK_SUCCESS);

        // acquire image from swapchain
        auto image_index = uint32_t();
        if(const auto ret = vkAcquireNextImageKHR(context.device.get(), context.swapchain.get(), uint64_max, context.image_avail_semaphores[current_frame].get(), VK_NULL_HANDLE, &image_index); ret == VK_ERROR_OUT_OF_DATE_KHR) {
            ensure(recreate_swapchain(window, context));
            continue;
        } else {
            ensure(ret == VK_SUCCESS || ret == VK_SUBOPTIMAL_KHR, "result={:x}", uint32_t(ret));
        }

        ensure(vkResetFences(context.device.get(), 1, &fence) == VK_SUCCESS);

        // update transform
        const auto time   = count / 60.0f;
        const auto aspect = 1.0f * context.swapchain_params.extent.width / context.swapchain_params.extent.height;

        auto ubo  = UniformBufferObject();
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view  = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj  = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        std::memcpy(uniform_buffers[current_frame].mapping.ptr, &ubo, sizeof(ubo));

        // fill command buffer
        ensure(vkResetCommandBuffer(context.command_buffers[current_frame], 0) == VK_SUCCESS);
        ensure(record_command_buffer({
            .pipeline        = context.pipeline.get(),
            .render_pass     = context.render_pass.get(),
            .vertex_buffer   = vertex_buffer.buffer.get(),
            .index_buffer    = index_buffer.buffer.get(),
            .framebuffer     = context.framebuffers[image_index].get(),
            .extent          = context.swapchain_params.extent,
            .command_buffer  = context.command_buffers[current_frame],
            .pipeline_layout = context.pipeline_layout.get(),
            .desc_set        = context.desc_sets[current_frame],
        }));
        // submit command
        const auto wait_semaphores   = std::array{context.image_avail_semaphores[current_frame].get()};
        const auto wait_stages       = std::array{VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)};
        const auto signal_semaphores = std::array{context.render_finished_semaphores[current_frame].get()};
        vk_args(vkQueueSubmit(graphics_queue, 1, &info, context.in_flight_fences[current_frame].get()),
                (VkSubmitInfo{
                    .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .waitSemaphoreCount   = 1,
                    .pWaitSemaphores      = wait_semaphores.data(),
                    .pWaitDstStageMask    = wait_stages.data(),
                    .commandBufferCount   = 1,
                    .pCommandBuffers      = &context.command_buffers[current_frame],
                    .signalSemaphoreCount = 1,
                    .pSignalSemaphores    = signal_semaphores.data(),
                }));

        // present rendered image
        const auto swapchains   = std::array{context.swapchain.get()};
        const auto present_info = VkPresentInfoKHR{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = signal_semaphores.data(),
            .swapchainCount     = 1,
            .pSwapchains        = swapchains.data(),
            .pImageIndices      = &image_index,
        };

        if(const auto ret = vkQueuePresentKHR(present_queue, &present_info); ret == VK_ERROR_OUT_OF_DATE_KHR || ret == VK_SUBOPTIMAL_KHR || context.window_resized) {
            ensure(recreate_swapchain(window, context));
            context.window_resized = false;
        } else {
            ensure(ret == VK_SUCCESS, "result={:x}", uint32_t(ret));
        }
        count += 1;
    }

    ensure(vkDeviceWaitIdle(context.device.get()) == VK_SUCCESS);

    return true;
}
} // namespace

auto main() -> int {
    ensure(glfwInit() == GLFW_TRUE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    unwrap_mut(window, glfwCreateWindow(800, 600, "vk", nullptr, nullptr));
    ensure(vulkan_main(window));
    glfwDestroyWindow(&window);
    glfwTerminate();
    return 0;
}
