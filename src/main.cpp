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
#include <unordered_map>
#include <vector>

//#define VULKAN_HPP_NO_EXCEPTIONS // TODO: Switch to a exception-free programm
#pragma warning(push)
#pragma warning(disable : 26812 28182)
#include <vulkan/vulkan.hpp>
#pragma warning(pop)

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#pragma warning(push)
#pragma warning(disable : 26451)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 6581 6606 26451 26495 26498)
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#pragma warning(pop)

constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

#if !defined(NDEBUG)
constexpr std::array validationLayers = {"VK_LAYER_KHRONOS_validation"};
#endif

constexpr std::array deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#if !defined(NDEBUG)
VkResult vkCreateDebugUtilsMessengerEXT(VkInstance instance,
                                        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                        const VkAllocationCallbacks* pAllocator,
                                        VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto vkCreateDebugUtilsMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (vkCreateDebugUtilsMessenger != nullptr)
        return vkCreateDebugUtilsMessenger(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void vkDestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                     const VkAllocationCallbacks* pAllocator) {
    auto vkDestroyDebugUtilsMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (vkDestroyDebugUtilsMessenger != nullptr)
        vkDestroyDebugUtilsMessenger(instance, debugMessenger, pAllocator);
}
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const noexcept {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;

    bool isAdequate() const noexcept { return !(formats.empty() || presentModes.empty()); }
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }

    static vk::VertexInputBindingDescription getBindingDescription() noexcept {
        vk::VertexInputBindingDescription bindingDescription;
        bindingDescription.setBinding(0);
        bindingDescription.setStride(sizeof(Vertex));
        bindingDescription.setInputRate(vk::VertexInputRate::eVertex);

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() noexcept {
        vk::VertexInputAttributeDescription posDesc;
        posDesc.setBinding(0);
        posDesc.setLocation(0);
        posDesc.setFormat(vk::Format::eR32G32B32Sfloat);
        posDesc.setOffset(offsetof(Vertex, pos));

        vk::VertexInputAttributeDescription colorDesc;
        colorDesc.setBinding(0);
        colorDesc.setLocation(1);
        colorDesc.setFormat(vk::Format::eR32G32B32Sfloat);
        colorDesc.setOffset(offsetof(Vertex, color));

        vk::VertexInputAttributeDescription texCoordDesc;
        texCoordDesc.setBinding(0);
        texCoordDesc.setLocation(2);
        texCoordDesc.setFormat(vk::Format::eR32G32Sfloat);
        texCoordDesc.setOffset(offsetof(Vertex, texCoord));

        return {posDesc, colorDesc, texCoordDesc};
    }
};

namespace std {
template <>
struct hash<Vertex> {
    size_t operator()(Vertex const& vertex) const {
        return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
               (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
};
}  // namespace std

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class VulkanGLFW {
   public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

   private:
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

    std::vector<Vertex> vertices;
    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;

    std::vector<uint32_t> indices;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;

    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;

    vk::Image textureImage;
    vk::DeviceMemory textureImageMemory;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

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

    void initWindow() noexcept {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void initVulkan() {
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
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void cleanupSwapChain() noexcept {
        device.destroyImageView(depthImageView);
        device.destroyImage(depthImage);
        device.freeMemory(depthImageMemory);

        for (auto& frameBuffer : swapChainFramebuffers) {
            device.destroyFramebuffer(frameBuffer);
        }

        device.freeCommandBuffers(commandPool, commandBuffers);

        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);

        for (auto imageView : swapChainImageViews) {
            device.destroyImageView(imageView);
        }

        device.destroySwapchainKHR(swapChain);

        for (auto& buffer : uniformBuffers) {
            device.destroyBuffer(buffer);
        }

        for (auto& memory : uniformBuffersMemory) {
            device.freeMemory(memory);
        }

        device.destroyDescriptorPool(descriptorPool);
    }

    void cleanup() noexcept {
        cleanupSwapChain();

        device.destroySampler(textureSampler);
        device.destroyImageView(textureImageView);
        device.destroyImage(textureImage);
        device.freeMemory(textureImageMemory);

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroyBuffer(vertexBuffer);
        device.freeMemory(vertexBufferMemory);

        device.destroyBuffer(indexBuffer);
        device.freeMemory(indexBufferMemory);

        for (auto& fence : inFlightFences) {
            device.destroyFence(fence);
        }

        for (auto& semaphore : renderFinishedSemaphores) {
            device.destroySemaphore(semaphore);
        }

        for (auto& semaphore : imageAvailableSemaphores) {
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

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    void createInstance() {
#if !defined(NDEBUG)
        // Check if validation layers are required and supported
        if (!checkValidationLayerSupport()) {
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

    constexpr vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo() {
        const auto severityFlags = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                   vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                   vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        const auto typeFlags = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                               vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                               vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

        return {{}, severityFlags, typeFlags, debugCallback};
    }

#if !defined(NDEBUG)
    void setupDebugMessenger() {
        const auto createInfo = getDebugMessengerCreateInfo();

        debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
    }
#endif

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr,
                                    reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        auto indices = findQueueFamilies(physicalDevice);
        std::set<uint32_t> uniqueQueueFamilies{indices.graphicsFamily.value(),
                                               indices.presentFamily.value()};

        constexpr float queuePriority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        for (auto queueFamily : uniqueQueueFamilies) {
            queueCreateInfos.push_back({{}, queueFamily, 1, &queuePriority});
        };

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.setSamplerAnisotropy(true);

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

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        const vk::SurfaceFormatKHR surfaceFormat =
            chooseSwapSurfaceFormat(swapChainSupport.formats);
        const vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        const vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                         indices.presentFamily.value()};

        vk::SharingMode imageSharingMode = vk::SharingMode::eExclusive;
        uint32_t queueFamilyIndexCount = 0;
        uint32_t* pQueueFamilyIndices = nullptr;
        if (indices.graphicsFamily != indices.presentFamily) {
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
        createInfo.setClipped(true);
        createInfo.setOldSwapchain(nullptr);

        swapChain = device.createSwapchainKHR(createInfo);

        swapChainImages = device.getSwapchainImagesKHR(swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat,
                                                     vk::ImageAspectFlagBits::eColor);
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment;
        colorAttachment.setFormat(swapChainImageFormat);
        colorAttachment.setSamples(vk::SampleCountFlagBits::e1);
        colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
        colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
        colorAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
        colorAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
        colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
        colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentDescription depthAttachment;
        depthAttachment.setFormat(findDepthFormat());
        depthAttachment.setSamples(vk::SampleCountFlagBits::e1);
        depthAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
        depthAttachment.setStoreOp(vk::AttachmentStoreOp::eDontCare);
        depthAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
        depthAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
        depthAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
        depthAttachment.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        std::array attachments = {colorAttachment, depthAttachment};

        vk::AttachmentReference colorAttachmentRef;
        colorAttachmentRef.setAttachment(0);
        colorAttachmentRef.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        vk::AttachmentReference depthAttachmentRef;
        depthAttachmentRef.setAttachment(1);
        depthAttachmentRef.setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::SubpassDescription subpass;
        subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
        subpass.setColorAttachmentCount(1);
        subpass.setPColorAttachments(&colorAttachmentRef);
        subpass.setPDepthStencilAttachment(&depthAttachmentRef);

        vk::SubpassDependency dependency;
        dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
        dependency.setDstSubpass(0);
        dependency.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        dependency.setSrcAccessMask({});
        dependency.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo renderPassInfo;
        renderPassInfo.setAttachmentCount(static_cast<uint32_t>(attachments.size()));
        renderPassInfo.setPAttachments(attachments.data());
        renderPassInfo.setSubpassCount(1);
        renderPassInfo.setPSubpasses(&subpass);
        renderPassInfo.setDependencyCount(1);
        renderPassInfo.setPDependencies(&dependency);

        renderPass = device.createRenderPass(renderPassInfo);
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding;
        uboLayoutBinding.setBinding(0);
        uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
        uboLayoutBinding.setDescriptorCount(1);
        uboLayoutBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
        uboLayoutBinding.setPImmutableSamplers(nullptr);

        vk::DescriptorSetLayoutBinding samplerLayoutBinding;
        samplerLayoutBinding.setBinding(1);
        samplerLayoutBinding.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
        samplerLayoutBinding.setDescriptorCount(1);
        samplerLayoutBinding.setStageFlags(vk::ShaderStageFlagBits::eFragment);
        samplerLayoutBinding.setPImmutableSamplers(nullptr);

        std::array bindings = {uboLayoutBinding, samplerLayoutBinding};

        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        layoutInfo.setBindingCount(static_cast<uint32_t>(bindings.size()));
        layoutInfo.setPBindings(bindings.data());

        descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
    }

    void createGraphicsPipeline() {
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

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                            fragShaderStageInfo};

        const auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
        vertexInputInfo.setVertexBindingDescriptionCount(1);
        vertexInputInfo.setPVertexBindingDescriptions(&bindingDescription);
        vertexInputInfo.setVertexAttributeDescriptionCount(
            static_cast<uint32_t>(attributeDescriptions.size()));
        vertexInputInfo.setPVertexAttributeDescriptions(attributeDescriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
        inputAssembly.setPrimitiveRestartEnable(false);

        const vk::Viewport viewport(0.f, 0.f, static_cast<float>(swapChainExtent.width),
                                    static_cast<float>(swapChainExtent.height), 0.f, 1.f);

        const vk::Rect2D scissor({0, 0}, swapChainExtent);

        vk::PipelineViewportStateCreateInfo viewportState;
        viewportState.setViewportCount(1);
        viewportState.setPViewports(&viewport);
        viewportState.setScissorCount(1);
        viewportState.setPScissors(&scissor);

        vk::PipelineRasterizationStateCreateInfo rasterizer;
        rasterizer.setDepthClampEnable(false);
        rasterizer.setPolygonMode(vk::PolygonMode::eFill);
        rasterizer.setLineWidth(1.f);
        rasterizer.setCullMode(vk::CullModeFlagBits::eBack);
        rasterizer.setFrontFace(vk::FrontFace::eCounterClockwise);
        rasterizer.setDepthBiasEnable(false);
        rasterizer.setDepthBiasConstantFactor(0.f);  // Optional
        rasterizer.setDepthBiasClamp(0.f);           // Optional
        rasterizer.setDepthBiasSlopeFactor(0.f);     // Optional

        vk::PipelineMultisampleStateCreateInfo multisampling;
        multisampling.setSampleShadingEnable(false);
        multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);
        multisampling.setMinSampleShading(1.0f);        // Optional
        multisampling.setPSampleMask(nullptr);          // Optional
        multisampling.setAlphaToCoverageEnable(false);  // Optional
        multisampling.setAlphaToOneEnable(false);       // Optional

        vk::PipelineDepthStencilStateCreateInfo depthStencil;
        depthStencil.setDepthTestEnable(true);
        depthStencil.setDepthWriteEnable(true);
        depthStencil.setDepthCompareOp(vk::CompareOp::eLess);
        depthStencil.setDepthBoundsTestEnable(false);
        depthStencil.setMinDepthBounds(0.0f);  // Optionnal
        depthStencil.setMaxDepthBounds(1.0f);  // Optionnal
        depthStencil.setStencilTestEnable(true);
        depthStencil.setFront({});  // Optionnal
        depthStencil.setBack({});   // Optionnal

        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.setColorWriteMask(
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        colorBlendAttachment.setBlendEnable(false);
        colorBlendAttachment.setSrcColorBlendFactor(vk::BlendFactor::eOne);   // Optional
        colorBlendAttachment.setDstColorBlendFactor(vk::BlendFactor::eZero);  // Optional
        colorBlendAttachment.setColorBlendOp(vk::BlendOp::eAdd);              // Optional
        colorBlendAttachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);   // Optional
        colorBlendAttachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero);  // Optional
        colorBlendAttachment.setAlphaBlendOp(vk::BlendOp::eAdd);              // Optional

        vk::PipelineColorBlendStateCreateInfo colorBlending;
        colorBlending.setLogicOpEnable(false);
        colorBlending.setLogicOp(vk::LogicOp::eCopy);  // Optional
        colorBlending.setAttachmentCount(1);
        colorBlending.setPAttachments(&colorBlendAttachment);
        colorBlending.setBlendConstants({0.f, 0.f, 0.f, 0.f});  // Optional

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setSetLayoutCount(1);
        pipelineLayoutInfo.setPSetLayouts(&descriptorSetLayout);
        pipelineLayoutInfo.setPushConstantRangeCount(0);     // Optional
        pipelineLayoutInfo.setPPushConstantRanges(nullptr);  // Optional

        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo;
        pipelineInfo.setStageCount(2);
        pipelineInfo.setPStages(shaderStages);
        pipelineInfo.setPVertexInputState(&vertexInputInfo);
        pipelineInfo.setPInputAssemblyState(&inputAssembly);
        pipelineInfo.setPViewportState(&viewportState);
        pipelineInfo.setPRasterizationState(&rasterizer);
        pipelineInfo.setPMultisampleState(&multisampling);
        pipelineInfo.setPDepthStencilState(&depthStencil);  // Optional
        pipelineInfo.setPColorBlendState(&colorBlending);
        pipelineInfo.setPDynamicState(nullptr);  // Optional
        pipelineInfo.setLayout(pipelineLayout);
        pipelineInfo.setRenderPass(renderPass);
        pipelineInfo.setSubpass(0);
        pipelineInfo.setBasePipelineHandle({});  // Optional
        pipelineInfo.setBasePipelineIndex(-1);   // Optional

        graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);

        device.destroyShaderModule(vertShaderModule);
        device.destroyShaderModule(fragShaderModule);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            const std::array attachments = {swapChainImageViews[i], depthImageView};

            vk::FramebufferCreateInfo framebufferInfo;
            framebufferInfo.setRenderPass(renderPass);
            framebufferInfo.setAttachmentCount(static_cast<uint32_t>(attachments.size()));
            framebufferInfo.setPAttachments(attachments.data());
            framebufferInfo.setWidth(swapChainExtent.width);
            framebufferInfo.setHeight(swapChainExtent.height);
            framebufferInfo.setLayers(1);

            swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        const vk::CommandPoolCreateInfo poolInfo({}, queueFamilyIndices.graphicsFamily.value());
        commandPool = device.createCommandPool(poolInfo);
    }

    void createDepthResources() {
        auto depthFormat = findDepthFormat();
        vk::Extent3D depthImageExtent(swapChainExtent, 1);

        std::tie(depthImage, depthImageMemory) =
            createImage(depthImageExtent, depthFormat, vk::ImageTiling::eOptimal,
                        vk::ImageUsageFlagBits::eDepthStencilAttachment,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);

        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);

        transitionImageLayout(depthImage, depthFormat, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        auto pixels =
            stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        auto imageSize = static_cast<vk::DeviceSize>(texWidth) * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        auto [stagingBuffer, stagingBufferMemory] = createBuffer(
            imageSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        void* data = device.mapMemory(stagingBufferMemory, 0, imageSize);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        device.unmapMemory(stagingBufferMemory);

        stbi_image_free(pixels);

        vk::Extent3D imageExtent;
        imageExtent.setWidth(static_cast<uint32_t>(texHeight));
        imageExtent.setHeight(static_cast<uint32_t>(texHeight));
        imageExtent.setDepth(1);

        std::tie(textureImage, textureImageMemory) =
            createImage(imageExtent, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);

        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eTransferDstOptimal);

        copyBufferToImage(stagingBuffer, textureImage, imageExtent);

        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb,
                              vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb,
                                           vk::ImageAspectFlagBits::eColor);
    }

    void createTextureSampler() {
        vk::SamplerCreateInfo samplerInfo;
        samplerInfo.setMagFilter(vk::Filter::eLinear);
        samplerInfo.setMinFilter(vk::Filter::eLinear);
        samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
        samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
        samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
        samplerInfo.setAnisotropyEnable(true);
        samplerInfo.setMaxAnisotropy(16.0f);
        samplerInfo.setBorderColor(vk::BorderColor::eIntOpaqueBlack);
        samplerInfo.setUnnormalizedCoordinates(false);
        samplerInfo.setCompareEnable(false);
        samplerInfo.setCompareOp(vk::CompareOp::eAlways);
        samplerInfo.setMipmapMode(vk::SamplerMipmapMode::eLinear);
        samplerInfo.setMipLodBias(0.0f);
        samplerInfo.setMinLod(0.0f);
        samplerInfo.setMaxLod(0.0f);

        textureSampler = device.createSampler(samplerInfo);
    }

    void createVertexBuffer() {
        const vk::DeviceSize bufferSize = sizeof(decltype(vertices)::value_type) * vertices.size();

        const vk::BufferUsageFlags stagingUsage = vk::BufferUsageFlagBits::eTransferSrc;
        const vk::MemoryPropertyFlags stagingProperties =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        auto [stagingBuffer, stagingBufferMemory] =
            createBuffer(bufferSize, stagingUsage, stagingProperties);

        void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        device.unmapMemory(stagingBufferMemory);

        const auto localUsage =
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
        const auto localProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;

        std::tie(vertexBuffer, vertexBufferMemory) =
            createBuffer(bufferSize, localUsage, localProperties);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex v{};
                v.pos = {attrib.vertices[static_cast<decltype(attrib.vertices)::size_type>(
                                             index.vertex_index) *
                                             3 +
                                         0],
                         attrib.vertices[static_cast<decltype(attrib.vertices)::size_type>(
                                             index.vertex_index) *
                                             3 +
                                         1],
                         attrib.vertices[static_cast<decltype(attrib.vertices)::size_type>(
                                             index.vertex_index) *
                                             3 +
                                         2]};
                v.texCoord = {
                    attrib.texcoords[static_cast<decltype(attrib.texcoords)::size_type>(
                                         index.texcoord_index) *
                                         2 +
                                     0],
                    1.0f - attrib.texcoords[static_cast<decltype(attrib.texcoords)::size_type>(
                                                index.texcoord_index) *
                                                2 +
                                            1]};
                v.color = {1.0f, 1.0f, 1.0f};

                if (uniqueVertices.count(v) == 0) {
                    uniqueVertices[v] =
                        static_cast<decltype(uniqueVertices)::mapped_type>(vertices.size());
                    vertices.push_back(v);
                }

                indices.push_back(uniqueVertices.at(v));
            }
        }
    }

    void createIndexBuffer() {
        const vk::DeviceSize bufferSize = sizeof(decltype(indices)::value_type) * indices.size();

        const auto stagingUsage = vk::BufferUsageFlagBits::eTransferSrc;
        const auto stagingProperties =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        auto [stagingBuffer, stagingBufferMemory] =
            createBuffer(bufferSize, stagingUsage, stagingProperties);

        void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        device.unmapMemory(stagingBufferMemory);

        const auto localUsage =
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;
        const auto localProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;

        std::tie(indexBuffer, indexBufferMemory) =
            createBuffer(bufferSize, localUsage, localProperties);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void createUniformBuffers() {
        constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        const auto usage = vk::BufferUsageFlagBits::eUniformBuffer;
        const auto properties =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            std::tie(uniformBuffers[i], uniformBuffersMemory[i]) =
                createBuffer(bufferSize, usage, properties);
        }
    }

    void createDescriptorPool() {
        vk::DescriptorPoolSize uniformPoolSize;
        uniformPoolSize.setType(vk::DescriptorType::eUniformBuffer);
        uniformPoolSize.setDescriptorCount(static_cast<uint32_t>(swapChainImages.size()));

        vk::DescriptorPoolSize combinedSamplerPoolSize;
        combinedSamplerPoolSize.setType(vk::DescriptorType::eCombinedImageSampler);
        combinedSamplerPoolSize.setDescriptorCount(static_cast<uint32_t>(swapChainImages.size()));

        std::array poolSizes{uniformPoolSize, combinedSamplerPoolSize};

        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()));
        poolInfo.setPPoolSizes(poolSizes.data());
        poolInfo.setMaxSets(static_cast<uint32_t>(swapChainImages.size()));

        descriptorPool = device.createDescriptorPool(poolInfo);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo;
        allocInfo.setDescriptorPool(descriptorPool);
        allocInfo.setDescriptorSetCount(static_cast<uint32_t>(swapChainImages.size()));
        allocInfo.setPSetLayouts(layouts.data());

        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo;
            bufferInfo.setBuffer(uniformBuffers[i]);
            bufferInfo.setOffset(0);
            bufferInfo.setRange(sizeof(UniformBufferObject));

            vk::DescriptorImageInfo imageInfo;
            imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
            imageInfo.setImageView(textureImageView);
            imageInfo.setSampler(textureSampler);

            vk::WriteDescriptorSet uniformDescriptorWrite;
            uniformDescriptorWrite.setDstSet(descriptorSets[i]);
            uniformDescriptorWrite.setDstBinding(0);
            uniformDescriptorWrite.setDstArrayElement(0);
            uniformDescriptorWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
            uniformDescriptorWrite.setDescriptorCount(1);
            uniformDescriptorWrite.setPBufferInfo(&bufferInfo);

            vk::WriteDescriptorSet combinedSamplerDescriptorWrite;
            combinedSamplerDescriptorWrite.setDstSet(descriptorSets[i]);
            combinedSamplerDescriptorWrite.setDstBinding(1);
            combinedSamplerDescriptorWrite.setDstArrayElement(0);
            combinedSamplerDescriptorWrite.setDescriptorType(
                vk::DescriptorType::eCombinedImageSampler);
            combinedSamplerDescriptorWrite.setDescriptorCount(1);
            combinedSamplerDescriptorWrite.setPImageInfo(&imageInfo);

            std::array descriptorWrites{uniformDescriptorWrite, combinedSamplerDescriptorWrite};

            device.updateDescriptorSets(descriptorWrites, nullptr);
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(swapChainFramebuffers.size());

        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setCommandPool(commandPool);
        allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
        allocInfo.setCommandBufferCount(static_cast<uint32_t>(commandBuffers.size()));

        commandBuffers = device.allocateCommandBuffers(allocInfo);

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            auto& cmdBuffer = commandBuffers[i];

            const vk::CommandBufferBeginInfo beginInfo({}, nullptr);

            cmdBuffer.begin(beginInfo);
            {
                std::array<vk::ClearValue, 2> clearValues{
                    vk::ClearColorValue(std::array<uint32_t, 4>{0, 0, 0, 1}),
                    vk::ClearDepthStencilValue(1, 0)};

                vk::RenderPassBeginInfo renderPassInfo;
                renderPassInfo.setRenderPass(renderPass);
                renderPassInfo.setFramebuffer(swapChainFramebuffers[i]);
                renderPassInfo.setRenderArea({{0, 0}, swapChainExtent});
                renderPassInfo.setClearValueCount(static_cast<uint32_t>(clearValues.size()));
                renderPassInfo.setPClearValues(clearValues.data());

                cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
                {
                    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
                    cmdBuffer.bindVertexBuffers(0, {vertexBuffer}, {0});
                    cmdBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
                    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
                                                 0, descriptorSets[i], nullptr);
                    cmdBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
                }
                cmdBuffer.endRenderPass();
            }
            cmdBuffer.end();
        }
    }

    void createSyncObjects() {
        imagesInFlight.resize(swapChainImages.size(), {});

        const vk::SemaphoreCreateInfo semaphoreInfo;
        const vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
            renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
            inFlightFences[i] = device.createFence(fenceInfo);
        }
    }

    void drawFrame() {
        device.waitForFences(inFlightFences[currentFrame], true, UINT64_MAX);

        vk::Result result;
        uint32_t imageIndex;
        std::tie(result, imageIndex) = device.acquireNextImageKHR(
            swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            device.waitForFences(imagesInFlight[imageIndex], true, UINT64_MAX);
        }

        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        updateUniformBuffer(imageIndex);

        const vk::Semaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        const vk::PipelineStageFlags waitStages[] = {
            vk::PipelineStageFlagBits::eColorAttachmentOutput};
        const vk::Semaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};

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

        const vk::SwapchainKHR swapChains[] = {swapChain};

        vk::PresentInfoKHR presentInfo;
        presentInfo.setWaitSemaphoreCount(1);
        presentInfo.setPWaitSemaphores(signalSemaphores);
        presentInfo.setSwapchainCount(1);
        presentInfo.setPSwapchains(swapChains);
        presentInfo.setPImageIndices(&imageIndex);
        presentInfo.setPResults(nullptr);  // Optional

        result = presentQueue.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR ||
            framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        presentQueue.waitIdle();

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    vk::Format findDepthFormat() {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
                                   vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
        for (auto format : candidates) {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear &&
                (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == vk::ImageTiling::eOptimal &&
                       (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    vk::ImageView createImageView(vk::Image& image, vk::Format format,
                                  vk::ImageAspectFlags aspectFlags) {
        vk::ImageSubresourceRange subresourceRange;
        subresourceRange.setAspectMask(aspectFlags);
        subresourceRange.setBaseMipLevel(0);
        subresourceRange.setLevelCount(1);
        subresourceRange.setBaseArrayLayer(0);
        subresourceRange.setLayerCount(1);

        vk::ImageViewCreateInfo viewInfo;
        viewInfo.setImage(image);
        viewInfo.setViewType(vk::ImageViewType::e2D);
        viewInfo.setFormat(format);
        viewInfo.setSubresourceRange(subresourceRange);

        return device.createImageView(viewInfo);
    }

    std::tuple<vk::Image, vk::DeviceMemory> createImage(vk::Extent3D& extent, vk::Format format,
                                                        vk::ImageTiling tiling,
                                                        vk::ImageUsageFlags usage,
                                                        vk::MemoryPropertyFlags properties) {
        vk::ImageCreateInfo imageInfo;
        imageInfo.setImageType(vk::ImageType::e2D);
        imageInfo.setExtent(extent);
        imageInfo.setMipLevels(1);
        imageInfo.setArrayLayers(1);
        imageInfo.setFormat(format);
        imageInfo.setTiling(tiling);
        imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
        imageInfo.setUsage(usage);
        imageInfo.setSharingMode(vk::SharingMode::eExclusive);
        imageInfo.setSamples(vk::SampleCountFlagBits::e1);

        auto image = device.createImage(imageInfo);

        const auto memRequirements = device.getImageMemoryRequirements(image);
        const auto memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        vk::MemoryAllocateInfo allocInfo;
        allocInfo.setAllocationSize(memRequirements.size);
        allocInfo.setMemoryTypeIndex(memoryTypeIndex);

        auto imageMemory = device.allocateMemory(allocInfo);

        device.bindImageMemory(image, imageMemory, 0);

        return {image, imageMemory};
    }

    void transitionImageLayout(vk::Image& image, vk::Format format, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout) {
        auto commandBuffer = beginSingleTimeCommands();

        vk::ImageSubresourceRange subresourceRange;
        subresourceRange.setBaseMipLevel(0);
        subresourceRange.setLevelCount(1);
        subresourceRange.setBaseArrayLayer(0);
        subresourceRange.setLayerCount(1);

        if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
            if (hasStencilComponent(format)) {
                subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
            }
        } else {
            subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        }

        vk::ImageMemoryBarrier barrier;
        barrier.setOldLayout(oldLayout);
        barrier.setNewLayout(newLayout);
        barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
        barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
        barrier.setImage(image);
        barrier.setSubresourceRange(subresourceRange);
        barrier.setSrcAccessMask({});
        barrier.setDstAccessMask({});

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.setSrcAccessMask({});
            barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
            barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
                   newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.setSrcAccessMask({});
            barrier.setDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                     vk::AccessFlagBits::eDepthStencilAttachmentWrite);

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, barrier);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(vk::Buffer& buffer, vk::Image& image, vk::Extent3D& extent) {
        auto commandBuffer = beginSingleTimeCommands();

        vk::ImageSubresourceLayers imgSubresource;
        imgSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
        imgSubresource.setMipLevel(0);
        imgSubresource.setBaseArrayLayer(0);
        imgSubresource.setLayerCount(1);

        vk::BufferImageCopy region;
        region.setBufferOffset(0);
        region.setBufferRowLength(0);
        region.setBufferImageHeight(0);
        region.setImageSubresource(imgSubresource);
        region.setImageOffset({0, 0, 0});
        region.setImageExtent(extent);

        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal,
                                        region);

        endSingleTimeCommands(commandBuffer);
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        const auto currentTime = std::chrono::high_resolution_clock::now();
        const float time =
            std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime)
                .count();

        UniformBufferObject ubo = {};
        ubo.model =
            glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(
            glm::radians(45.0f), swapChainExtent.width / static_cast<float>(swapChainExtent.height),
            0.1f, 10.0f);

        ubo.proj[1][1] *= -1;  // invert Y-axis (OpenGL -> Vulkan)

        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    std::tuple<vk::Buffer, vk::DeviceMemory> createBuffer(vk::DeviceSize size,
                                                          vk::BufferUsageFlags usage,
                                                          vk::MemoryPropertyFlags properties) {
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

        return {buffer, bufferMemory};
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
        auto commandBuffer = beginSingleTimeCommands();

        vk::BufferCopy copyRegion;
        copyRegion.setSrcOffset(0);
        copyRegion.setDstOffset(0);
        copyRegion.setSize(size);

        commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    vk::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
        allocInfo.setCommandPool(commandPool);
        allocInfo.setCommandBufferCount(1);

        vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

        const vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(vk::CommandBuffer& commandBuffer) {
        commandBuffer.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1);
        submitInfo.setPCommandBuffers(&commandBuffer);

        graphicQueue.submit(submitInfo, nullptr);
        graphicQueue.waitIdle();

        device.freeCommandBuffers(commandPool, commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        const auto memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    vk::ShaderModule createShaderModule(const std::vector<char>& code) {
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCodeSize(code.size());
        createInfo.setPCode(reinterpret_cast<const uint32_t*>(code.data()));

        return device.createShaderModule(createInfo);
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& availableFormats) noexcept {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(
        const std::vector<vk::PresentModeKHR>& availablePresentModes) noexcept {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                std::cout << "Mailbox" << std::endl;
                return availablePresentMode;
            }
            if (availablePresentMode == vk::PresentModeKHR::eFifoRelaxed) {
                std::cout << "Fifo Relaxed" << std::endl;
                return availablePresentMode;
            }
            if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                std::cout << "Immediate" << std::endl;
                return availablePresentMode;
            }
        }

        std::cout << "Fifo" << std::endl;
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) noexcept {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                                            capabilities.maxImageExtent.width);
            actualExtent.height =
                std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                           capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device) {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
        details.formats = device.getSurfaceFormatsKHR(surface);
        details.presentModes = device.getSurfacePresentModesKHR(surface);

        return details;
    }

    bool isDeviceSuitable(const vk::PhysicalDevice& device) {
        const auto indices = findQueueFamilies(device);
        const auto extensionsSupported = checkDeviceExtensionSupport(device);

        auto swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = swapChainSupport.isAdequate();
        }

        const auto supportedFeatures = device.getFeatures();

        return indices.isComplete() && extensionsSupported && swapChainAdequate &&
               supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) {
        auto availableExtensions = device.enumerateDeviceExtensionProperties();

        for (auto extensionName : deviceExtensions) {
            bool extensionFound = false;

            for (auto& extensionProperty : availableExtensions) {
                if (strncmp(extensionName, extensionProperty.extensionName,
                            std::size(extensionProperty.extensionName)) == 0) {
                    extensionFound = true;
                    break;
                }
            }

            if (!extensionFound) {
                return false;
            }
        }

        return true;
    }

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) {
        QueueFamilyIndices indices;

        auto queueFamilies = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            if (device.getSurfaceSupportKHR(i, surface)) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#if !defined(NDEBUG)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

        return extensions;
    }

    std::vector<const char*> getAvailableExtensions() {
        std::vector<vk::ExtensionProperties> extensions =
            vk::enumerateInstanceExtensionProperties();

        std::cout << "available extensions:" << std::endl;

        for (const auto& extension : extensions) {
            std::cout << "\t" << extension.extensionName << std::endl;
        }
    }

#if !defined(NDEBUG)
    bool checkValidationLayerSupport() {
        auto availableLayers = vk::enumerateInstanceLayerProperties();

        for (auto layerName : validationLayers) {
            bool layerFound = false;

            for (auto& layerPropertie : availableLayers) {
                if (strncmp(layerName, layerPropertie.layerName,
                            std::size(layerPropertie.layerName)) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }
#endif

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        const auto fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) noexcept {
        auto app = static_cast<VulkanGLFW*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

int main() {
    VulkanGLFW app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}