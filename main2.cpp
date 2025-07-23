#include <cstring>
#include <ranges>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "macros/unwrap.hpp"
#include "vk.hpp"

namespace {
struct Vertex {
    glm::vec2 pos;
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
};

const auto vertices = std::array{
    Vertex{{0.0, -0.5}, {1, 0, 0}},
    Vertex{{0.5, 0.5}, {0, 1, 0}},
    Vertex{{-0.5, 0.5}, {0, 0, 1}},
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
    const auto device_features    = VkPhysicalDeviceFeatures{};
    const auto device_create_info = VkDeviceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount    = uint32_t(queue_create_infos.size()),
        .pQueueCreateInfos       = queue_create_infos.data(),
        .enabledExtensionCount   = required_exts.size(),
        .ppEnabledExtensionNames = required_exts.data(),
        .pEnabledFeatures        = &device_features,
    };

    ensure(vkCreateDevice(phy, &device_create_info, nullptr, &vk::default_device) == VK_SUCCESS);
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

auto create_image_views(VkDevice device, VkSwapchainKHR swapchain, const VkFormat& swapchain_format) -> std::optional<std::vector<vk::AutoVkImageView>> {
    // retrieve images from swapchain
    unwrap(swapchain_images, vk::query_array<VkImage>([&](auto... args) { return vkGetSwapchainImagesKHR(device, swapchain, args...); }));
    PRINT("images={}", swapchain_images.size());

    // create image views
    auto image_views = std::vector<vk::AutoVkImageView>(swapchain_images.size());
    for(auto&& [image, view] : std::ranges::zip_view(swapchain_images, image_views)) {
        const auto create_info = VkImageViewCreateInfo{
            .sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image      = image,
            .viewType   = VK_IMAGE_VIEW_TYPE_2D,
            .format     = swapchain_format,
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
        };
        ensure(vkCreateImageView(device, &create_info, nullptr, std::inout_ptr(view)) == VK_SUCCESS);
    }
    return image_views;
}

auto create_pipeline_layout(VkDevice device) -> VkPipelineLayout_T* {
    auto pipeline_layout_create_info = VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    auto pipeline_layout = VkPipelineLayout();
    ensure(vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipeline_layout) == VK_SUCCESS);
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

    const auto render_pass_create_info = VkRenderPassCreateInfo{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &color_attachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &subpass_dep,
    };
    auto render_pass = VkRenderPass();
    ensure(vkCreateRenderPass(device, &render_pass_create_info, nullptr, &render_pass) == VK_SUCCESS);
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
        .frontFace               = VK_FRONT_FACE_CLOCKWISE,
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

    const auto pipeline_create_info = VkGraphicsPipelineCreateInfo{
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
    };
    auto pipeline = VkPipeline();
    ensure(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline) == VK_SUCCESS);
    return pipeline;
}

auto create_framebuffers(VkDevice device, std::span<const vk::AutoVkImageView> image_views, VkRenderPass render_pass, VkExtent2D swapchain_extent) -> std::optional<std::vector<vk::AutoVkFramebuffer>> {
    auto framebuffers = std::vector<vk::AutoVkFramebuffer>(image_views.size());
    for(auto&& [fb, image] : std::ranges::zip_view(framebuffers, image_views)) {
        const auto attachments = std::array{image.get()};
        const auto create_info = VkFramebufferCreateInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = render_pass,
            .attachmentCount = uint32_t(attachments.size()),
            .pAttachments    = attachments.data(),
            .width           = swapchain_extent.width,
            .height          = swapchain_extent.height,
            .layers          = 1,
        };
        ensure(vkCreateFramebuffer(device, &create_info, nullptr, std::inout_ptr(fb)) == VK_SUCCESS);
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

auto create_buffer(VkPhysicalDevice phy, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) -> std::optional<std::pair<vk::AutoVkBuffer, vk::AutoVkDeviceMemory>> {
    // create buffer
    const auto buffer_create_info = VkBufferCreateInfo{
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    auto buffer = vk::AutoVkBuffer();
    ensure(vkCreateBuffer(device, &buffer_create_info, nullptr, std::inout_ptr(buffer)) == VK_SUCCESS);

    // allocate memory
    auto requirements = VkMemoryRequirements();
    vkGetBufferMemoryRequirements(device, buffer.get(), &requirements);

    unwrap(memory_type, find_memory_type(phy, requirements.memoryTypeBits, props));
    auto alloc_info = VkMemoryAllocateInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = requirements.size,
        .memoryTypeIndex = memory_type,
    };
    auto memory = vk::AutoVkDeviceMemory();
    ensure(vkAllocateMemory(device, &alloc_info, nullptr, std::inout_ptr(memory)) == VK_SUCCESS);

    // bind them
    ensure(vkBindBufferMemory(device, buffer.get(), memory.get(), 0) == VK_SUCCESS);

    return std::make_pair(std::move(buffer), std::move(memory));
}

auto create_command_pool(VkDevice device, uint32_t graphics_queue_index) -> VkCommandPool_T* {
    const auto command_pool_create_info = VkCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = uint32_t(graphics_queue_index),
    };
    auto command_pool = VkCommandPool();
    ensure(vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool) == VK_SUCCESS);
    return command_pool;
}

auto allocate_command_buffers(VkDevice device, VkCommandPool command_pool, uint32_t count) -> std::optional<std::vector<VkCommandBuffer>> {
    const auto command_buffer_alloc_info = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count,
    };
    auto command_buffers = std::vector<VkCommandBuffer>(count);
    ensure(vkAllocateCommandBuffers(device, &command_buffer_alloc_info, command_buffers.data()) == VK_SUCCESS);
    return command_buffers;
}

auto create_semaphores(VkDevice device, uint32_t count) -> std::optional<std::vector<vk::AutoVkSemaphore>> {
    const auto semaphore_create_info = VkSemaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    auto ret = std::vector<vk::AutoVkSemaphore>(count);
    for(auto& r : ret) {
        ensure(vkCreateSemaphore(device, &semaphore_create_info, nullptr, std::inout_ptr(r)) == VK_SUCCESS);
    }
    return ret;
}

auto create_fences(VkDevice device, uint32_t count, bool signaled) -> std::optional<std::vector<vk::AutoVkFence>> {
    const auto fence_create_info = VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : VkFenceCreateFlagBits(0),
    };
    auto ret = std::vector<vk::AutoVkFence>(count);
    for(auto& r : ret) {
        ensure(vkCreateFence(device, &fence_create_info, nullptr, std::inout_ptr(r)) == VK_SUCCESS);
    }
    return ret;
}

auto record_command_buffer(VkPipeline pipeline, VkRenderPass render_pass, VkBuffer vertex_buffer, VkFramebuffer framebuffer, VkExtent2D extent, VkCommandBuffer command_buffer, uint32_t index) -> bool {
    const auto command_buffer_info = VkCommandBufferBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
    };
    ensure(vkBeginCommandBuffer(command_buffer, &command_buffer_info) == VK_SUCCESS);
    const auto clear_color      = VkClearValue{.color = {.float32 = {0, 0, 0, 1}}};
    const auto render_pass_info = VkRenderPassBeginInfo{
        .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass  = render_pass,
        .framebuffer = framebuffer,
        .renderArea  = {
             .offset = {0, 0},
             .extent = extent,
        },
        .clearValueCount = 1,
        .pClearValues    = &clear_color,
    };
    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const auto vertex_buffers = std::array{vertex_buffer};
    const auto offsets        = std::array{VkDeviceSize(0)};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers.data(), offsets.data());
    const auto viewport = VkViewport{
        .x        = 0,
        .y        = 0,
        .width    = float(extent.width),
        .height   = float(extent.height),
        .minDepth = 0,
        .maxDepth = 1,
    };
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    const auto scissor = VkRect2D{
        .offset = {0, 0},
        .extent = extent,
    };
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);
    vkCmdDraw(command_buffer, vertices.size(), 1, 0, 0);
    vkCmdEndRenderPass(command_buffer);
    ensure(vkEndCommandBuffer(command_buffer) == VK_SUCCESS);
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
    vk::AutoVkPipelineLayout           pipeline_layout;
    vk::AutoVkRenderPass               render_pass;
    vk::AutoVkPipeline                 pipeline;
    std::vector<vk::AutoVkFramebuffer> framebuffers;
    vk::AutoVkBuffer                   vertex_buffer;
    vk::AutoVkDeviceMemory             vertex_memory;
    vk::AutoVkCommandPool              command_pool;
    std::vector<VkCommandBuffer>       command_buffers;
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
        unwrap_mut(pipeline_layout, create_pipeline_layout(context.device.get()));
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
    {
        unwrap_mut(vertex_buffer, create_buffer(context.phy, context.device.get(), sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
        context.vertex_buffer = std::move(vertex_buffer.first);
        context.vertex_memory = std::move(vertex_buffer.second);
    }
    {
        const auto size = sizeof(Vertex) * vertices.size();
        unwrap_mut(mapping, vk::MemoryMapping::map(context.device.get(), context.vertex_memory.get(), size));
        std::memcpy(mapping.ptr, vertices.data(), size);
    }
    {
        unwrap_mut(command_pool, create_command_pool(context.device.get(), context.graphics_queue_index));
        context.command_pool.reset(&command_pool);
    }
    constexpr auto max_frames_in_flight = 2;
    {
        unwrap_mut(command_buffers, allocate_command_buffers(context.device.get(), context.command_pool.get(), max_frames_in_flight));
        context.command_buffers = std::move(command_buffers);
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
    auto current_frame = uint32_t(0);

    while(!glfwWindowShouldClose(&window)) {
        glfwPollEvents();

        constexpr auto uint64_max = std::numeric_limits<uint64_t>::max();

        // wait for previous command completion
        const auto fence = context.in_flight_fences[current_frame].get();
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

        // fill command buffer
        ensure(vkResetCommandBuffer(context.command_buffers[current_frame], 0) == VK_SUCCESS);
        ensure(record_command_buffer(context.pipeline.get(), context.render_pass.get(), context.vertex_buffer.get(), context.framebuffers[image_index].get(), context.swapchain_params.extent, context.command_buffers[current_frame], image_index));
        // submit command
        const auto wait_semaphores   = std::array{context.image_avail_semaphores[current_frame].get()};
        const auto wait_stages       = std::array{VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)};
        const auto signal_semaphores = std::array{context.render_finished_semaphores[current_frame].get()};
        const auto submit_info       = VkSubmitInfo{
                  .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                  .waitSemaphoreCount   = 1,
                  .pWaitSemaphores      = wait_semaphores.data(),
                  .pWaitDstStageMask    = wait_stages.data(),
                  .commandBufferCount   = 1,
                  .pCommandBuffers      = &context.command_buffers[current_frame],
                  .signalSemaphoreCount = 1,
                  .pSignalSemaphores    = signal_semaphores.data(),
        };
        ensure(vkQueueSubmit(graphics_queue, 1, &submit_info, context.in_flight_fences[current_frame].get()) == VK_SUCCESS);

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
        current_frame = (current_frame + 1) % max_frames_in_flight;
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
