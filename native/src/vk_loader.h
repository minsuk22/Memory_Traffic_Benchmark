// Minimal Vulkan function loader (no SDK / loader import library needed).
// Works with Android's libvulkan.so, Linux libvulkan.so.1 and Windows vulkan-1.dll.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#define MB_VK_GLOBAL_FUNCS(X)              \
    X(vkCreateInstance)                    \
    X(vkEnumerateInstanceVersion)          \
    X(vkEnumerateInstanceExtensionProperties)

#define MB_VK_INSTANCE_FUNCS(X)                  \
    X(vkDestroyInstance)                         \
    X(vkEnumeratePhysicalDevices)                \
    X(vkGetPhysicalDeviceProperties)             \
    X(vkGetPhysicalDeviceProperties2)            \
    X(vkGetPhysicalDeviceFeatures2)              \
    X(vkGetPhysicalDeviceQueueFamilyProperties)  \
    X(vkGetPhysicalDeviceMemoryProperties)       \
    X(vkEnumerateDeviceExtensionProperties)      \
    X(vkCreateDevice)                            \
    X(vkGetDeviceProcAddr)

#define MB_VK_DEVICE_FUNCS(X)          \
    X(vkDestroyDevice)                 \
    X(vkGetDeviceQueue)                \
    X(vkCreateBuffer)                  \
    X(vkCreateBufferView)              \
    X(vkDestroyBufferView)             \
    X(vkDestroyBuffer)                 \
    X(vkGetBufferMemoryRequirements)   \
    X(vkAllocateMemory)                \
    X(vkFreeMemory)                    \
    X(vkBindBufferMemory)              \
    X(vkMapMemory)                     \
    X(vkUnmapMemory)                   \
    X(vkCreateShaderModule)            \
    X(vkDestroyShaderModule)           \
    X(vkCreateDescriptorSetLayout)     \
    X(vkDestroyDescriptorSetLayout)    \
    X(vkCreatePipelineLayout)          \
    X(vkDestroyPipelineLayout)         \
    X(vkCreateComputePipelines)        \
    X(vkDestroyPipeline)               \
    X(vkCreateDescriptorPool)          \
    X(vkDestroyDescriptorPool)         \
    X(vkAllocateDescriptorSets)        \
    X(vkFreeDescriptorSets)            \
    X(vkUpdateDescriptorSets)          \
    X(vkCreateCommandPool)             \
    X(vkDestroyCommandPool)            \
    X(vkAllocateCommandBuffers)        \
    X(vkFreeCommandBuffers)            \
    X(vkResetCommandBuffer)            \
    X(vkBeginCommandBuffer)            \
    X(vkEndCommandBuffer)              \
    X(vkCmdBindPipeline)               \
    X(vkCmdBindDescriptorSets)         \
    X(vkCmdPushConstants)              \
    X(vkCmdDispatch)                   \
    X(vkCmdFillBuffer)                 \
    X(vkCmdPipelineBarrier)            \
    X(vkCmdWriteTimestamp)             \
    X(vkCmdResetQueryPool)             \
    X(vkCreateQueryPool)               \
    X(vkDestroyQueryPool)              \
    X(vkGetQueryPoolResults)           \
    X(vkCreateFence)                   \
    X(vkDestroyFence)                  \
    X(vkResetFences)                   \
    X(vkWaitForFences)                 \
    X(vkQueueSubmit)                   \
    X(vkDeviceWaitIdle)

// Optional (VK_KHR_cooperative_matrix): may stay null.
#define MB_VK_OPTIONAL_INSTANCE_FUNCS(X) \
    X(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)

// Optional (VK_KHR_pipeline_executable_properties): may stay null.
#define MB_VK_OPTIONAL_DEVICE_FUNCS(X)          \
    X(vkGetPipelineExecutablePropertiesKHR)     \
    X(vkGetPipelineExecutableStatisticsKHR)

#define MB_VK_DECLARE(name) extern PFN_##name name;
MB_VK_GLOBAL_FUNCS(MB_VK_DECLARE)
MB_VK_INSTANCE_FUNCS(MB_VK_DECLARE)
MB_VK_DEVICE_FUNCS(MB_VK_DECLARE)
MB_VK_OPTIONAL_DEVICE_FUNCS(MB_VK_DECLARE)
MB_VK_OPTIONAL_INSTANCE_FUNCS(MB_VK_DECLARE)
#undef MB_VK_DECLARE

namespace vkl {
// Each returns false and fills `err` with the first missing symbol on failure.
bool loadLibrary(const char** err);
bool loadInstance(VkInstance instance, const char** err);
bool loadDevice(VkDevice device, const char** err);
}  // namespace vkl
