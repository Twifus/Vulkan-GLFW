#ifdef _WIN32
#define NOMINMAX
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

//#define VULKAN_HPP_NO_EXCEPTIONS // TODO: Switch to a exception-free programm
#pragma warning(push)
#pragma warning(disable : 26812 28182)
#include <vulkan/vulkan.hpp>
#pragma warning(pop)

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

#if !defined(NDEBUG)
constexpr std::array validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
#endif

constexpr std::array deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#if !defined(NDEBUG)
VkResult vkCreateDebugUtilsMessengerEXT(VkInstance instance,
                                        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                        const VkAllocationCallbacks* pAllocator,
                                        VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto vkCreateDebugUtilsMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (vkCreateDebugUtilsMessenger != nullptr)
        return vkCreateDebugUtilsMessenger(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void vkDestroyDebugUtilsMessengerEXT(VkInstance instance,
                                     VkDebugUtilsMessengerEXT debugMessenger,
                                     const VkAllocationCallbacks* pAllocator)
{
    auto vkDestroyDebugUtilsMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (vkDestroyDebugUtilsMessenger != nullptr)
        vkDestroyDebugUtilsMessenger(instance, debugMessenger, pAllocator);
}
#endif

class VulkanGLFW
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() const noexcept
        {
            return graphicsFamily.has_value()
                && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails
    {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;

        bool isAdequate() const noexcept
        {
            return !(formats.empty() || presentModes.empty());
        }
    };

    struct Vertex
    {
        glm::vec2 pos;
        glm::vec3 color;

        static vk::VertexInputBindingDescription getBindingDescription() noexcept
        {
            vk::VertexInputBindingDescription bindingDescription;
            bindingDescription.setBinding(0);
            bindingDescription.setStride(sizeof(Vertex));
            bindingDescription.setInputRate(vk::VertexInputRate::eVertex);

            return bindingDescription;
        }

        static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() noexcept
        {
            vk::VertexInputAttributeDescription posDesc;
            posDesc.setBinding(0);
            posDesc.setLocation(0);
            posDesc.setFormat(vk::Format::eR32G32Sfloat);
            posDesc.setOffset(offsetof(Vertex, pos));

            vk::VertexInputAttributeDescription colorDesc;
            colorDesc.setBinding(0);
            colorDesc.setLocation(1);
            colorDesc.setFormat(vk::Format::eR32G32B32Sfloat);
            colorDesc.setOffset(offsetof(Vertex, color));

            return { posDesc, colorDesc };
        }
    };

    struct UniformBufferObject
    {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
    };

    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 3, 0
    };

    GLFWwindow* window = nullptr;

    vk::Instance instance;
    #if !defined(NDEBUG)
    vk::DebugUtilsMessengerEXT debugMessenger;
    #endif
    vk::SurfaceKHR surface;

    vk::PhysicalDevice physicalDevice;
    vk::Device device;

    vk::Queue graphicQueue;
    vk::Queue presentQueue;

    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    vk::RenderPass renderPass;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;
    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;

    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> imageAvailableSemaphores;
    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> renderFinishedSemaphores;
    std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow() noexcept
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void initVulkan()
    {
        createInstance();
        #if !defined(NDEBUG)
        setupDebugMessenger();
        #endif
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void cleanupSwapChain() noexcept
    {
        for (auto& frameBuffer : swapChainFramebuffers)
        {
            device.destroyFramebuffer(frameBuffer);
        }

        device.freeCommandBuffers(commandPool, commandBuffers);

        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);

        for (auto imageView : swapChainImageViews)
        {
            device.destroyImageView(imageView);
        }

        device.destroySwapchainKHR(swapChain);

        for (auto& buffer : uniformBuffers)
        {
            device.destroyBuffer(buffer);
        }

        for (auto& memory : uniformBuffersMemory)
        {
            device.freeMemory(memory);
        }

        device.destroyDescriptorPool(descriptorPool);
    }

    void cleanup() noexcept
    {
        cleanupSwapChain();

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroyBuffer(vertexBuffer);
        device.freeMemory(vertexBufferMemory);

        device.destroyBuffer(indexBuffer);
        device.freeMemory(indexBufferMemory);

        for (auto& fence : inFlightFences)
        {
            device.destroyFence(fence);
        }

        for (auto& semaphore : renderFinishedSemaphores)
        {
            device.destroySemaphore(semaphore);
        }

        for (auto& semaphore : imageAvailableSemaphores)
        {
            device.destroySemaphore(semaphore);
        }

        device.destroyCommandPool(commandPool);

        device.destroy();

        #if !defined(NDEBUG)
        instance.destroyDebugUtilsMessengerEXT(debugMessenger);
        #endif

        instance.destroySurfaceKHR(surface);
        instance.destroy();

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    void createInstance()
    {
        #if !defined(NDEBUG)
        // Check if validation layers are required and supported
        if (!checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        #endif

        // Create Application Info
        vk::ApplicationInfo appInfo;
        appInfo.setPApplicationName("HelloTriangle");
        appInfo.setApplicationVersion(VK_MAKE_VERSION(1, 0, 0));
        appInfo.setPEngineName("TooGoodEngine");
        appInfo.setEngineVersion(VK_MAKE_VERSION(1, 0, 0));
        appInfo.setApiVersion(VK_API_VERSION_1_2);

        // Get required extensions
        auto extensions = getRequiredExtensions();

        // Create Instance Info
        vk::InstanceCreateInfo createInfo;
        createInfo.setPApplicationInfo(&appInfo);
        createInfo.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()));
        createInfo.setPpEnabledExtensionNames(extensions.data());

        #if !defined(NDEBUG)
        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        debugCreateInfo = getDebugMessengerCreateInfo();
        createInfo.pNext = &debugCreateInfo;
        #endif

        // Create Instance
        instance = vk::createInstance(createInfo);
    }

    constexpr vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo()
    {
        const auto severityFlags = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        const auto typeFlags = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
            | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

        return { {}, severityFlags, typeFlags, debugCallback };
    }

    #if !defined(NDEBUG)
    void setupDebugMessenger()
    {
        const auto createInfo = getDebugMessengerCreateInfo();

        debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
    }
    #endif

    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice()
    {
        std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        if (devices.empty())
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice()
    {
        auto indices = findQueueFamilies(physicalDevice);
        std::set<uint32_t> uniqueQueueFamilies{ indices.graphicsFamily.value(), indices.presentFamily.value() };

        constexpr float queuePriority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        for (auto queueFamily : uniqueQueueFamilies)
        {
            queueCreateInfos.push_back({ {}, queueFamily, 1, &queuePriority });
        };

        const vk::PhysicalDeviceFeatures deviceFeatures;

        vk::DeviceCreateInfo createInfo;
        createInfo.setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()));
        createInfo.setPQueueCreateInfos(queueCreateInfos.data());
        createInfo.setEnabledExtensionCount(static_cast<uint32_t>(deviceExtensions.size()));
        createInfo.setPpEnabledExtensionNames(deviceExtensions.data());
        createInfo.setPEnabledFeatures(&deviceFeatures);

        #if !defined(NDEBUG)
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        #endif

        device = physicalDevice.createDevice(createInfo);

        graphicQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        const vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        const vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        const vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        vk::SharingMode imageSharingMode = vk::SharingMode::eExclusive;
        uint32_t queueFamilyIndexCount = 0;
        uint32_t* pQueueFamilyIndices = nullptr;
        if (indices.graphicsFamily != indices.presentFamily)
        {
            imageSharingMode = vk::SharingMode::eConcurrent;
            queueFamilyIndexCount = 2;
            pQueueFamilyIndices = queueFamilyIndices;
        }

        vk::SwapchainCreateInfoKHR createInfo;
        createInfo.setSurface(surface);
        createInfo.setMinImageCount(imageCount);
        createInfo.setImageFormat(surfaceFormat.format);
        createInfo.setImageColorSpace(surfaceFormat.colorSpace);
        createInfo.setImageExtent(extent);
        createInfo.setImageArrayLayers(1);
        createInfo.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
        createInfo.setImageSharingMode(imageSharingMode);
        createInfo.setQueueFamilyIndexCount(queueFamilyIndexCount);
        createInfo.setPQueueFamilyIndices(pQueueFamilyIndices);
        createInfo.setPreTransform(swapChainSupport.capabilities.currentTransform);
        createInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
        createInfo.setPresentMode(presentMode);
        createInfo.setClipped(VK_TRUE);
        createInfo.setOldSwapchain(nullptr);

        swapChain = device.createSwapchainKHR(createInfo);

        swapChainImages = device.getSwapchainImagesKHR(swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            const vk::ComponentMapping components; // identity for all components

            vk::ImageSubresourceRange subresourceRange;
            subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
            subresourceRange.setBaseMipLevel(0);
            subresourceRange.setLevelCount(1);
            subresourceRange.setBaseArrayLayer(0);
            subresourceRange.setLayerCount(1);

            vk::ImageViewCreateInfo createInfo;
            createInfo.setImage(swapChainImages[i]);
            createInfo.setViewType(vk::ImageViewType::e2D);
            createInfo.setFormat(swapChainImageFormat);
            createInfo.setComponents(components);
            createInfo.setSubresourceRange(subresourceRange);

            swapChainImageViews[i] = device.createImageView(createInfo);
        }
    }

    void createRenderPass()
    {
        vk::AttachmentDescription colorAttachment;
        colorAttachment.setFormat(swapChainImageFormat);
        colorAttachment.setSamples(vk::SampleCountFlagBits::e1);
        colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
        colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
        colorAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
        colorAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
        colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
        colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentReference colorAttachmentRef;
        colorAttachmentRef.setAttachment(0);
        colorAttachmentRef.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass;
        subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
        subpass.setColorAttachmentCount(1);
        subpass.setPColorAttachments(&colorAttachmentRef);

        vk::SubpassDependency dependency;
        dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
        dependency.setDstSubpass(0);
        dependency.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        dependency.setSrcAccessMask({});
        dependency.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo renderPassInfo;
        renderPassInfo.setAttachmentCount(1);
        renderPassInfo.setPAttachments(&colorAttachment);
        renderPassInfo.setSubpassCount(1);
        renderPassInfo.setPSubpasses(&subpass);
        renderPassInfo.setDependencyCount(1);
        renderPassInfo.setPDependencies(&dependency);

        renderPass = device.createRenderPass(renderPassInfo);
    }

    void createDescriptorSetLayout()
    {
        vk::DescriptorSetLayoutBinding uboLayoutBinding;
        uboLayoutBinding.setBinding(0);
        uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
        uboLayoutBinding.setDescriptorCount(1);
        uboLayoutBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
        uboLayoutBinding.setPImmutableSamplers(nullptr);

        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        layoutInfo.setBindingCount(1);
        layoutInfo.setPBindings(&uboLayoutBinding);

        descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        const auto vertShaderModule = createShaderModule(vertShaderCode);
        const auto fragShaderModule = createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
        vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex);
        vertShaderStageInfo.setModule(vertShaderModule);
        vertShaderStageInfo.setPName("main");

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
        fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
        fragShaderStageInfo.setModule(fragShaderModule);
        fragShaderStageInfo.setPName("main");

        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        const auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
        vertexInputInfo.setVertexBindingDescriptionCount(1);
        vertexInputInfo.setPVertexBindingDescriptions(&bindingDescription);
        vertexInputInfo.setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()));
        vertexInputInfo.setPVertexAttributeDescriptions(attributeDescriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
        inputAssembly.setPrimitiveRestartEnable(VK_FALSE);

        const vk::Viewport viewport(0.f, 0.f,
                                    static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height),
                                    0.f, 1.f);

        const vk::Rect2D scissor({ 0, 0 },
                                 swapChainExtent);

        vk::PipelineViewportStateCreateInfo viewportState;
        viewportState.setViewportCount(1);
        viewportState.setPViewports(&viewport);
        viewportState.setScissorCount(1);
        viewportState.setPScissors(&scissor);

        vk::PipelineRasterizationStateCreateInfo rasterizer;
        rasterizer.setDepthClampEnable(VK_FALSE);
        rasterizer.setPolygonMode(vk::PolygonMode::eFill);
        rasterizer.setLineWidth(1.f);
        rasterizer.setCullMode(vk::CullModeFlagBits::eBack);
        rasterizer.setFrontFace(vk::FrontFace::eCounterClockwise);
        rasterizer.setDepthBiasEnable(VK_FALSE);
        rasterizer.setDepthBiasConstantFactor(0.f); // Optional
        rasterizer.setDepthBiasClamp(0.f); // Optional
        rasterizer.setDepthBiasSlopeFactor(0.f); // Optional

        vk::PipelineMultisampleStateCreateInfo multisampling;
        multisampling.setSampleShadingEnable(VK_FALSE);
        multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);
        multisampling.setMinSampleShading(1.0f); // Optional
        multisampling.setPSampleMask(nullptr); // Optional
        multisampling.setAlphaToCoverageEnable(VK_FALSE); // Optional
        multisampling.setAlphaToOneEnable(VK_FALSE); // Optional

        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        colorBlendAttachment.setBlendEnable(VK_FALSE);
        colorBlendAttachment.setSrcColorBlendFactor(vk::BlendFactor::eOne); // Optional
        colorBlendAttachment.setDstColorBlendFactor(vk::BlendFactor::eZero); // Optional
        colorBlendAttachment.setColorBlendOp(vk::BlendOp::eAdd); // Optional
        colorBlendAttachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne); // Optional
        colorBlendAttachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero); // Optional
        colorBlendAttachment.setAlphaBlendOp(vk::BlendOp::eAdd); // Optional

        vk::PipelineColorBlendStateCreateInfo colorBlending;
        colorBlending.setLogicOpEnable(VK_FALSE);
        colorBlending.setLogicOp(vk::LogicOp::eCopy); // Optional
        colorBlending.setAttachmentCount(1);
        colorBlending.setPAttachments(&colorBlendAttachment);
        colorBlending.setBlendConstants({ 0.f, 0.f, 0.f, 0.f }); // Optional

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setSetLayoutCount(1);
        pipelineLayoutInfo.setPSetLayouts(&descriptorSetLayout);
        pipelineLayoutInfo.setPushConstantRangeCount(0); // Optional
        pipelineLayoutInfo.setPPushConstantRanges(nullptr); // Optional

        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo;
        pipelineInfo.setStageCount(2);
        pipelineInfo.setPStages(shaderStages);
        pipelineInfo.setPVertexInputState(&vertexInputInfo);
        pipelineInfo.setPInputAssemblyState(&inputAssembly);
        pipelineInfo.setPViewportState(&viewportState);
        pipelineInfo.setPRasterizationState(&rasterizer);
        pipelineInfo.setPMultisampleState(&multisampling);
        pipelineInfo.setPDepthStencilState(nullptr); // Optional
        pipelineInfo.setPColorBlendState(&colorBlending);
        pipelineInfo.setPDynamicState(nullptr); // Optional
        pipelineInfo.setLayout(pipelineLayout);
        pipelineInfo.setRenderPass(renderPass);
        pipelineInfo.setSubpass(0);
        pipelineInfo.setBasePipelineHandle({}); // Optional
        pipelineInfo.setBasePipelineIndex(-1); // Optional

        graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);

        device.destroyShaderModule(vertShaderModule);
        device.destroyShaderModule(fragShaderModule);
    }

    void createVertexBuffer()
    {
        const vk::DeviceSize bufferSize = sizeof(decltype(vertices)::value_type) * vertices.size();

        const vk::BufferUsageFlags stagingUsage = vk::BufferUsageFlagBits::eTransferSrc;
        const vk::MemoryPropertyFlags stagingProperties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, stagingUsage, stagingProperties);

        void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        device.unmapMemory(stagingBufferMemory);

        const auto localUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
        const auto localProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;

        std::tie(vertexBuffer, vertexBufferMemory) = createBuffer(bufferSize, localUsage, localProperties);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void createIndexBuffer()
    {
        const vk::DeviceSize bufferSize = sizeof(decltype(indices)::value_type) * indices.size();

        const auto stagingUsage = vk::BufferUsageFlagBits::eTransferSrc;
        const auto stagingProperties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, stagingUsage, stagingProperties);

        void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        device.unmapMemory(stagingBufferMemory);

        const auto localUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;
        const auto localProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;

        std::tie(indexBuffer, indexBufferMemory) = createBuffer(bufferSize, localUsage, localProperties);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void createUniformBuffers()
    {
        constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        const auto usage = vk::BufferUsageFlagBits::eUniformBuffer;
        const auto properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            std::tie(uniformBuffers[i], uniformBuffersMemory[i]) = createBuffer(bufferSize, usage, properties);
        }
    }

    void createDescriptorPool()
    {
        const vk::DescriptorPoolSize poolSize(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(swapChainImages.size()));

        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setPoolSizeCount(1);
        poolInfo.setPPoolSizes(&poolSize);
        poolInfo.setMaxSets(static_cast<uint32_t>(swapChainImages.size()));

        descriptorPool = device.createDescriptorPool(poolInfo);
    }

    void createDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo;
        allocInfo.setDescriptorPool(descriptorPool);
        allocInfo.setDescriptorSetCount(static_cast<uint32_t>(swapChainImages.size()));
        allocInfo.setPSetLayouts(layouts.data());

        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            const vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));
            vk::WriteDescriptorSet descriptorWrite;
            descriptorWrite.setDstSet(descriptorSets[i]);
            descriptorWrite.setDstBinding(0);
            descriptorWrite.setDstArrayElement(0);
            descriptorWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
            descriptorWrite.setDescriptorCount(1);
            descriptorWrite.setPBufferInfo(&bufferInfo);

            device.updateDescriptorSets(descriptorWrite, nullptr);
        }
    }

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            const vk::ImageView attachments[] = {
                swapChainImageViews[i]
            };

            vk::FramebufferCreateInfo framebufferInfo;
            framebufferInfo.setRenderPass(renderPass);
            framebufferInfo.setAttachmentCount(1);
            framebufferInfo.setPAttachments(attachments);
            framebufferInfo.setWidth(swapChainExtent.width);
            framebufferInfo.setHeight(swapChainExtent.height);
            framebufferInfo.setLayers(1);

            swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);

        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        const vk::CommandPoolCreateInfo poolInfo({}, queueFamilyIndices.graphicsFamily.value());
        commandPool = device.createCommandPool(poolInfo);
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(swapChainFramebuffers.size());

        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setCommandPool(commandPool);
        allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
        allocInfo.setCommandBufferCount(static_cast<uint32_t>(commandBuffers.size()));

        commandBuffers = device.allocateCommandBuffers(allocInfo);

        for (size_t i = 0; i < commandBuffers.size(); i++)
        {
            auto& cmdBuffer = commandBuffers[i];

            const vk::CommandBufferBeginInfo beginInfo({}, nullptr);

            cmdBuffer.begin(beginInfo);
            {
                vk::ClearValue clearColor({ std::array<float, 4>{ 0.f, 0.f, 0.f, 1.f } });

                vk::RenderPassBeginInfo renderPassInfo;
                renderPassInfo.setRenderPass(renderPass);
                renderPassInfo.setFramebuffer(swapChainFramebuffers[i]);
                renderPassInfo.setRenderArea({ {0, 0}, swapChainExtent });
                renderPassInfo.setClearValueCount(1);
                renderPassInfo.setPClearValues(&clearColor);

                cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
                {
                    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
                    cmdBuffer.bindVertexBuffers(0, { vertexBuffer }, { 0 });
                    cmdBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
                    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets[i], nullptr);
                    cmdBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
                }
                cmdBuffer.endRenderPass();
            }
            cmdBuffer.end();
        }
    }

    void createSyncObjects()
    {
        imagesInFlight.resize(swapChainImages.size(), {});

        const vk::SemaphoreCreateInfo semaphoreInfo;
        const vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
            renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
            inFlightFences[i] = device.createFence(fenceInfo);
        }
    }

    void drawFrame()
    {
        device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        vk::Result result;
        uint32_t imageIndex;
        std::tie(result, imageIndex) = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            recreateSwapChain();
            return;
        }
        else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            device.waitForFences(imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }

        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        updateUniformBuffer(imageIndex);

        const vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        const vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        const vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };

        vk::SubmitInfo submitInfo;
        submitInfo.setWaitSemaphoreCount(1);
        submitInfo.setPWaitSemaphores(waitSemaphores);
        submitInfo.setPWaitDstStageMask(waitStages);
        submitInfo.setCommandBufferCount(1);
        submitInfo.setPCommandBuffers(&commandBuffers[imageIndex]);
        submitInfo.setSignalSemaphoreCount(1);
        submitInfo.setPSignalSemaphores(signalSemaphores);

        device.resetFences(inFlightFences[currentFrame]);

        graphicQueue.submit(submitInfo, inFlightFences[currentFrame]);

        const vk::SwapchainKHR swapChains[] = { swapChain };

        vk::PresentInfoKHR presentInfo;
        presentInfo.setWaitSemaphoreCount(1);
        presentInfo.setPWaitSemaphores(signalSemaphores);
        presentInfo.setSwapchainCount(1);
        presentInfo.setPSwapchains(swapChains);
        presentInfo.setPImageIndices(&imageIndex);
        presentInfo.setPResults(nullptr); // Optional

        result = presentQueue.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        presentQueue.waitIdle();

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        const auto currentTime = std::chrono::high_resolution_clock::now();
        const float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);

        ubo.proj[1][1] *= -1; // invert Y-axis (OpenGL -> Vulkan)

        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    std::tuple<vk::Buffer, vk::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
    {
        vk::BufferCreateInfo bufferInfo;
        bufferInfo.setSize(size);
        bufferInfo.setUsage(usage);
        bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

        auto buffer = device.createBuffer(bufferInfo);

        const auto memRequirements = device.getBufferMemoryRequirements(buffer);
        const auto memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        vk::MemoryAllocateInfo allocInfo;
        allocInfo.setAllocationSize(memRequirements.size);
        allocInfo.setMemoryTypeIndex(memoryTypeIndex);

        auto bufferMemory = device.allocateMemory(allocInfo);
        device.bindBufferMemory(buffer, bufferMemory, 0);

        return { buffer, bufferMemory };
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
    {
        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
        allocInfo.setCommandPool(commandPool);
        allocInfo.setCommandBufferCount(1);

        vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

        const vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        commandBuffer.begin(beginInfo);
        {
            vk::BufferCopy copyRegion;
            copyRegion.setSrcOffset(0);
            copyRegion.setDstOffset(0);
            copyRegion.setSize(size);

            commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
        }
        commandBuffer.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1);
        submitInfo.setPCommandBuffers(&commandBuffer);

        graphicQueue.submit(submitInfo, nullptr);
        graphicQueue.waitIdle();

        device.freeCommandBuffers(commandPool, commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        const auto memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    vk::ShaderModule createShaderModule(const std::vector<char>& code)
    {
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCodeSize(code.size());
        createInfo.setPCode(reinterpret_cast<const uint32_t*>(code.data()));

        return device.createShaderModule(createInfo);
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) noexcept
    {
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) noexcept
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox)
            {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) noexcept
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent(static_cast<uint32_t>(width),
                                      static_cast<uint32_t>(height));
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device)
    {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
        details.formats = device.getSurfaceFormatsKHR(surface);
        details.presentModes = device.getSurfacePresentModesKHR(surface);

        return details;
    }

    bool isDeviceSuitable(const vk::PhysicalDevice& device)
    {
        const auto indices = findQueueFamilies(device);
        const auto extensionsSupported = checkDeviceExtensionSupport(device);

        auto swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = swapChainSupport.isAdequate();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device)
    {
        auto availableExtensions = device.enumerateDeviceExtensionProperties();

        for (auto extensionName : deviceExtensions)
        {
            bool extensionFound = false;

            for (auto& extensionProperty : availableExtensions)
            {
                if (strncmp(extensionName, extensionProperty.extensionName, std::size(extensionProperty.extensionName)) == 0)
                {
                    extensionFound = true;
                    break;
                }
            }

            if (!extensionFound)
            {
                return false;
            }
        }

        return true;
    }

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device)
    {
        QueueFamilyIndices indices;

        auto queueFamilies = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.graphicsFamily = i;
            }

            if (device.getSurfaceSupportKHR(i, surface))
            {
                indices.presentFamily = i;
            }

            if (indices.isComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        #if !defined(NDEBUG)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        #endif

        return extensions;
    }

    std::vector<const char*> getAvailableExtensions()
    {
        std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

        std::cout << "available extensions:" << std::endl;

        for (const auto& extension : extensions)
        {
            std::cout << "\t" << extension.extensionName << std::endl;
        }
    }

    #if !defined(NDEBUG)
    bool checkValidationLayerSupport()
    {
        auto availableLayers = vk::enumerateInstanceLayerProperties();

        for (auto layerName : validationLayers)
        {
            bool layerFound = false;

            for (auto& layerPropertie : availableLayers)
            {
                if (strncmp(layerName, layerPropertie.layerName, std::size(layerPropertie.layerName)) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }
    #endif

    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file!");
        }

        const auto fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) noexcept
    {
        auto app = static_cast<VulkanGLFW*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

int main()
{
    VulkanGLFW app;

    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}