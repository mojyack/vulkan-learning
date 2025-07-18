#include <ranges>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include "macros/unwrap.hpp"
#include "vk.hpp"

namespace {
auto vulkan_main(GLFWwindow& window) -> bool {
    unwrap(exts, vk::query_array<VkExtensionProperties>([](auto... args) { return vkEnumerateInstanceExtensionProperties(nullptr, args...); }));
    PRINT("exts={}", exts.size());

    // init vulkan
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

    auto instance = vk::AutoVkInstance();
    ensure(vkCreateInstance(&instance_create_info, nullptr, std::inout_ptr(instance)) == VK_SUCCESS);
    vk::default_instance = instance.get();

    // create window surface
    auto surface = vk::AutoVkSurface();
    ensure(glfwCreateWindowSurface(instance.get(), &window, nullptr, std::inout_ptr(surface)) == VK_SUCCESS);

    // pickup phy
    static const auto required_exts = std::array{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    unwrap(devices, vk::query_array<VkPhysicalDevice>([instance = instance.get()](auto... args) { return vkEnumeratePhysicalDevices(instance, args...); }));
    ensure(!devices.empty());

    auto phy = VkPhysicalDevice(VK_NULL_HANDLE);
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
        unwrap(sc, vk::SwapchainDetail::query(dev, surface.get()));
        if(sc.formats.empty() || sc.modes.empty()) {
            continue;
        }
        phy = dev;
        break;
    }
    ensure(phy != VK_NULL_HANDLE);

    // setup queue
    unwrap(queue_families, vk::query_array<VkQueueFamilyProperties>([phy](auto... args) {vkGetPhysicalDeviceQueueFamilyProperties(phy, args...); return VK_SUCCESS; }));
    PRINT("queues={}", queue_families.size());
    auto graphics_queue_index = -1;
    auto present_queue_index  = -1;
    for(auto i = 0uz; i < queue_families.size(); i += 1) {
        const auto& family = queue_families[i];

        auto support_present = VkBool32(false);
        ensure(vkGetPhysicalDeviceSurfaceSupportKHR(phy, i, surface.get(), &support_present) == VK_SUCCESS);

        std::println("queue {}: flags={:02x} graphics?={} present?={}", i, family.queueFlags, family.queueFlags & VK_QUEUE_GRAPHICS_BIT, support_present);
        if(family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphics_queue_index = i;
        }
        if(support_present) {
            present_queue_index = i;
        }
    }
    ensure(graphics_queue_index >= 0);

    // create logical device
    auto queue_create_infos = std::vector<VkDeviceQueueCreateInfo>();
    for(auto i : {graphics_queue_index, present_queue_index}) {
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

    auto device = vk::AutoVkDevice();
    ensure(vkCreateDevice(phy, &device_create_info, nullptr, std::inout_ptr(device)) == VK_SUCCESS);
    PRINT("logical device created");
    vk::default_device = device.get();

    // retrieve queue handle
    auto graphics_queue = VkQueue();
    auto present_queue  = VkQueue();
    vkGetDeviceQueue(device.get(), graphics_queue_index, 0, &graphics_queue);
    vkGetDeviceQueue(device.get(), present_queue_index, 0, &present_queue);

    // decide swapchain params
    unwrap(swapchain_detail, vk::SwapchainDetail::query(phy, surface.get()));
    auto swapchain_format = swapchain_detail.formats[0];
    for(const auto& format : swapchain_detail.formats) {
        if(format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            swapchain_format = format;
            break;
        }
    }
    auto swapchain_mode   = VK_PRESENT_MODE_FIFO_KHR;
    auto swapchain_extent = swapchain_detail.caps.currentExtent;
    if(swapchain_extent.width == std::numeric_limits<uint32_t>::max()) {
        auto fb = std::array<int, 2>();
        glfwGetFramebufferSize(&window, &fb[0], &fb[1]);
        const auto caps         = swapchain_detail.caps;
        swapchain_extent.width  = std::clamp<uint32_t>(fb[0], caps.minImageExtent.width, caps.maxImageExtent.width);
        swapchain_extent.height = std::clamp<uint32_t>(fb[1], caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    auto swapchain_image_count = swapchain_detail.caps.minImageCount + 1;
    if(swapchain_detail.caps.maxImageCount != 0 && swapchain_image_count > swapchain_detail.caps.maxImageCount) {
        swapchain_image_count = swapchain_detail.caps.maxImageCount;
    }
    PRINT("extent={}x{} images={}~{}", swapchain_extent.width, swapchain_extent.height, swapchain_detail.caps.minImageCount, swapchain_detail.caps.maxImageCount);

    // create swapchain
    auto swapchain_create_info = VkSwapchainCreateInfoKHR{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = surface.get(),
        .minImageCount    = swapchain_image_count,
        .imageFormat      = swapchain_format.format,
        .imageColorSpace  = swapchain_format.colorSpace,
        .imageExtent      = swapchain_extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = swapchain_detail.caps.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = swapchain_mode,
        .clipped          = VK_TRUE,
    };
    auto queue_indices = std::array{uint32_t(graphics_queue_index), uint32_t(present_queue_index)};
    if(graphics_queue_index != present_queue_index) {
        swapchain_create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        swapchain_create_info.queueFamilyIndexCount = queue_indices.size();
        swapchain_create_info.pQueueFamilyIndices   = queue_indices.data();
    } else {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    auto swapchain = vk::AutoVkSwapchain();
    ensure(vkCreateSwapchainKHR(device.get(), &swapchain_create_info, nullptr, std::inout_ptr(swapchain)) == VK_SUCCESS);
    PRINT("swapchain created");

    // retrieve images from swapchain
    unwrap(swapchain_images, vk::query_array<VkImage>([&](auto... args) { return vkGetSwapchainImagesKHR(device.get(), swapchain.get(), args...); }));
    PRINT("images={}", swapchain_images.size());

    // create image views
    auto image_views = std::vector<vk::AutoVkImageView>(swapchain_images.size());
    for(auto&& [image, view] : std::ranges::zip_view(swapchain_images, image_views)) {
        const auto create_info = VkImageViewCreateInfo{
            .sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image      = image,
            .viewType   = VK_IMAGE_VIEW_TYPE_2D,
            .format     = swapchain_format.format,
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
        ensure(vkCreateImageView(device.get(), &create_info, nullptr, std::inout_ptr(view)) == VK_SUCCESS);
    }

    // create shaders
    auto vert_shader = vk::AutoVkShaderModule(vk::create_shader_module(device.get(), "build/vert.spv"));
    ensure(vert_shader);
    auto frag_shader = vk::AutoVkShaderModule(vk::create_shader_module(device.get(), "build/frag.spv"));
    ensure(frag_shader);

    const auto shader_stages = std::array<VkPipelineShaderStageCreateInfo, 2>{{
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_shader.get(),
            .pName  = "main",
        },
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_shader.get(),
            .pName  = "main",
        },
    }};

    // vertex input
    const auto vertex_input_state_create_info = VkPipelineVertexInputStateCreateInfo{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = 0,
        .pVertexBindingDescriptions      = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions    = nullptr,
    };
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
    // pipeline layout
    auto pipeline_layout_create_info = VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    auto pipeline_layout = vk::AutoVkPipelineLayout();
    ensure(vkCreatePipelineLayout(device.get(), &pipeline_layout_create_info, nullptr, std::inout_ptr(pipeline_layout)) == VK_SUCCESS);

    // attachments
    const auto color_attachment = VkAttachmentDescription{
        .format         = swapchain_format.format,
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
    const auto subpass_dep = VkSubpassDependency{
        .srcSubpass    = VK_SUBPASS_EXTERNAL,
        .dstSubpass    = 0,
        .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    // render pass
    const auto render_pass_create_info = VkRenderPassCreateInfo{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &color_attachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &subpass_dep,
    };
    auto render_pass = vk::AutoVkRenderPass();
    ensure(vkCreateRenderPass(device.get(), &render_pass_create_info, nullptr, std::inout_ptr(render_pass)) == VK_SUCCESS);

    // pipeline
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
        .layout              = pipeline_layout.get(),
        .renderPass          = render_pass.get(),
        .subpass             = 0,
    };
    auto pipeline = vk::AutoVkPipeline();
    ensure(vkCreateGraphicsPipelines(device.get(), VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, std::inout_ptr(pipeline)) == VK_SUCCESS);

    // wrap swapchain images to framebuffers
    auto framebuffers = std::vector<vk::AutoVkFramebuffer>(image_views.size());
    for(auto&& [fb, image] : std::ranges::zip_view(framebuffers, image_views)) {
        const auto attachments = std::array{image.get()};
        const auto create_info = VkFramebufferCreateInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = render_pass.get(),
            .attachmentCount = uint32_t(attachments.size()),
            .pAttachments    = attachments.data(),
            .width           = swapchain_extent.width,
            .height          = swapchain_extent.height,
            .layers          = 1,
        };
        ensure(vkCreateFramebuffer(device.get(), &create_info, nullptr, std::inout_ptr(fb)) == VK_SUCCESS);
    }

    // command pool
    constexpr auto max_frames_in_flight = 2;

    const auto command_pool_create_info = VkCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = uint32_t(graphics_queue_index),
    };
    auto command_pool = vk::AutoVkCommandPool();
    ensure(vkCreateCommandPool(device.get(), &command_pool_create_info, nullptr, std::inout_ptr(command_pool)) == VK_SUCCESS);
    // command buffer
    const auto command_buffer_alloc_info = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = command_pool.get(),
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = max_frames_in_flight,
    };
    auto command_buffers = std::array<VkCommandBuffer, max_frames_in_flight>();
    ensure(vkAllocateCommandBuffers(device.get(), &command_buffer_alloc_info, command_buffers.data()) == VK_SUCCESS);

    const auto record_command_buffer = [&](VkCommandBuffer command_buffer, uint32_t index) -> bool {
        constexpr auto error_value = false;

        const auto command_buffer_info = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
        };
        ensure_v(vkBeginCommandBuffer(command_buffer, &command_buffer_info) == VK_SUCCESS);
        const auto clear_color      = VkClearValue{.color = {.float32 = {0, 0, 0, 1}}};
        const auto render_pass_info = VkRenderPassBeginInfo{
            .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass  = render_pass.get(),
            .framebuffer = framebuffers[index].get(),
            .renderArea  = {
                 .offset = {0, 0},
                 .extent = swapchain_extent,
            },
            .clearValueCount = 1,
            .pClearValues    = &clear_color,
        };
        vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.get());
        const auto viewport = VkViewport{
            .x        = 0,
            .y        = 0,
            .width    = float(swapchain_extent.width),
            .height   = float(swapchain_extent.height),
            .minDepth = 0,
            .maxDepth = 1,
        };
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        const auto scissor = VkRect2D{
            .offset = {0, 0},
            .extent = swapchain_extent,
        };
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(command_buffer);
        ensure_v(vkEndCommandBuffer(command_buffer) == VK_SUCCESS);
        return true;
    };

    // sync primitives
    const auto semaphore_create_info = VkSemaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    const auto fence_create_info = VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    auto image_avail_semaphores     = std::array<vk::AutoVkSemaphore, max_frames_in_flight>();
    auto render_finished_semaphores = std::array<vk::AutoVkSemaphore, max_frames_in_flight>();
    auto in_flight_fences           = std::array<vk::AutoVkFence, max_frames_in_flight>();
    for(auto i = 0; i < max_frames_in_flight; i += 1) {
        ensure(vkCreateSemaphore(device.get(), &semaphore_create_info, nullptr, std::inout_ptr(image_avail_semaphores[i])) == VK_SUCCESS);
        ensure(vkCreateSemaphore(device.get(), &semaphore_create_info, nullptr, std::inout_ptr(render_finished_semaphores[i])) == VK_SUCCESS);
        ensure(vkCreateFence(device.get(), &fence_create_info, nullptr, std::inout_ptr(in_flight_fences[i])) == VK_SUCCESS);
    }

    auto current_frame = uint32_t(0);
    while(!glfwWindowShouldClose(&window)) {
        glfwPollEvents();

        constexpr auto uint64_max = std::numeric_limits<uint64_t>::max();

        // wait for previous command completion
        const auto fence = in_flight_fences[current_frame].get();
        ensure(vkWaitForFences(device.get(), 1, &fence, VK_TRUE, uint64_max) == VK_SUCCESS);
        ensure(vkResetFences(device.get(), 1, &fence) == VK_SUCCESS);
        // acquire image from swapchain
        auto image_index = uint32_t();
        ensure(vkAcquireNextImageKHR(device.get(), swapchain.get(), uint64_max, image_avail_semaphores[current_frame].get(), VK_NULL_HANDLE, &image_index) == VK_SUCCESS || true /* TODO: handle error*/);
        PRINT("image={}", image_index);
        // fill command buffer
        ensure(vkResetCommandBuffer(command_buffers[current_frame], 0) == VK_SUCCESS);
        ensure(record_command_buffer(command_buffers[current_frame], image_index));
        // submit command
        const auto wait_semaphores   = std::array{image_avail_semaphores[current_frame].get()};
        const auto wait_stages       = std::array{VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)};
        const auto signal_semaphores = std::array{render_finished_semaphores[current_frame].get()};
        const auto submit_info       = VkSubmitInfo{
                  .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                  .waitSemaphoreCount   = 1,
                  .pWaitSemaphores      = wait_semaphores.data(),
                  .pWaitDstStageMask    = wait_stages.data(),
                  .commandBufferCount   = 1,
                  .pCommandBuffers      = &command_buffers[current_frame],
                  .signalSemaphoreCount = 1,
                  .pSignalSemaphores    = signal_semaphores.data(),
        };
        ensure(vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame].get()) == VK_SUCCESS);
        // present rendered image
        const auto swapchains   = std::array{swapchain.get()};
        const auto present_info = VkPresentInfoKHR{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = signal_semaphores.data(),
            .swapchainCount     = 1,
            .pSwapchains        = swapchains.data(),
            .pImageIndices      = &image_index,
        };
        ensure(vkQueuePresentKHR(present_queue, &present_info) == VK_SUCCESS);
        current_frame = (current_frame + 1) % max_frames_in_flight;
    }

    ensure(vkDeviceWaitIdle(device.get()) == VK_SUCCESS);

    return true;
}
} // namespace

auto main() -> int {
    ensure(glfwInit() == GLFW_TRUE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    unwrap_mut(window, glfwCreateWindow(800, 600, "vk", nullptr, nullptr));
    ensure(vulkan_main(window));
    glfwDestroyWindow(&window);
    glfwTerminate();
    return 0;
}
