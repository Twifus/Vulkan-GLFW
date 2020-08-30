# VulkanGLFW

Simple model viewer program written in C++17.

## Dependencies

- [GLFW3](https://www.glfw.org/)
- [Vulkan](https://www.khronos.org/vulkan/)
- [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp)
- [GLM](https://glm.g-truc.net/)
- [stb](https://github.com/nothings/stb)
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)

## Progress

- [x] Simple GLFW window
- [x] Vulkan basic init
- [x] Pipeline Setup
- [x] Vertex Buffers
- [x] Uniform Buffers
- [x] Texture mapping
- [x] Depth Buffer
- [x] Model loading
- [ ] Memory Management
- [ ] Exception-free

## Plans

- Considere [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) for better memory management
- Use [EASTL](https://github.com/electronicarts/EASTL) ?
- [IMGUI](https://github.com/ocornut/imgui) for on-screen menu
- switch from glm to google's [mathfu](https://github.com/google/mathfu)
- [glTF](https://www.khronos.org/gltf/) ?