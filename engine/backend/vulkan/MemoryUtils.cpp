module;

module VulkanBackend.Vulkan.MemoryUtils;

import std;
import std.compat;

import vulkan;

namespace VulkanBackend::Vulkan {

namespace {
std::optional<std::uint32_t> FindMemoryTypeInternal(vk::PhysicalDevice physical_device,
                                               std::uint32_t type_filter,
                                               vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties mem_properties = physical_device.getMemoryProperties();

    for (std::uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return std::nullopt;
}

std::optional<std::uint32_t> TryFindMemoryType(vk::PhysicalDevice physical_device,
                                          std::uint32_t type_filter,
                                          vk::MemoryPropertyFlags required,
                                          vk::MemoryPropertyFlags preferred) {
    if (auto index = FindMemoryTypeInternal(physical_device, type_filter, required | preferred)) {
        return index;
    }
    if (preferred != vk::MemoryPropertyFlags{}) {
        if (auto index = FindMemoryTypeInternal(physical_device, type_filter, required | preferred)) {
            return index;
        }
    }
    if (required != vk::MemoryPropertyFlags{}) {
        if (auto index = FindMemoryTypeInternal(physical_device, type_filter, required)) {
            return index;
        }
    }
    return std::nullopt;
}

} // namespace

uint32_t MemoryUtils::FindMemoryType(vk::PhysicalDevice physical_device,
                                     std::uint32_t type_filter,
                                     vk::MemoryPropertyFlags properties) {
    if (auto index = FindMemoryTypeInternal(physical_device, type_filter, properties)) {
        return *index;
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

bool MemoryUtils::GetMemoryBudget(vk::PhysicalDevice physical_device,
                                  std::vector<vk::DeviceSize>& heap_budgets,
                                  std::vector<vk::DeviceSize>& heap_usages) {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget_props{};
    vk::PhysicalDeviceMemoryProperties2 properties{};
    properties.pNext = &budget_props;

    physical_device.getMemoryProperties2(&properties);

    const std::uint32_t heap_count = properties.memoryProperties.memoryHeapCount;
    heap_budgets.assign(budget_props.heapBudget.begin(), budget_props.heapBudget.begin() + heap_count);
    heap_usages.assign(budget_props.heapUsage.begin(), budget_props.heapUsage.begin() + heap_count);

    const bool has_non_zero_budget = std::ranges::any_of(heap_budgets, [](vk::DeviceSize value) {
        return value != 0;
    });

    const bool has_usage_data = std::ranges::any_of(heap_usages, [](vk::DeviceSize value) {
        return value != 0;
    });

    return has_non_zero_budget || has_usage_data;
}

vk::DeviceSize MemoryUtils::AlignedSize(vk::DeviceSize size, vk::DeviceSize alignment) {
    if (alignment == 0) {
        return size;
    }
    const vk::DeviceSize remainder = size % alignment;
    if (remainder == 0) {
        return size;
    }
    return size + alignment - remainder;
}

vk::MemoryPropertyFlags MemoryUtils::GetMemoryTypeProperties(vk::PhysicalDevice physical_device,
                                                          std::uint32_t memory_type_index) {
    vk::PhysicalDeviceMemoryProperties mem_properties = physical_device.getMemoryProperties();
    if (memory_type_index >= mem_properties.memoryTypeCount) {
        throw std::out_of_range("Memory type index out of range");
    }
    return mem_properties.memoryTypes[memory_type_index].propertyFlags;
}

bool MemoryUtils::IsMemoryTypeHostVisible(vk::PhysicalDevice physical_device, std::uint32_t memory_type_index) {
    const vk::MemoryPropertyFlags flags = GetMemoryTypeProperties(physical_device, memory_type_index);
    return (flags & vk::MemoryPropertyFlagBits::eHostVisible) != vk::MemoryPropertyFlags{};
}

bool MemoryUtils::IsMemoryTypeDeviceLocal(vk::PhysicalDevice physical_device, std::uint32_t memory_type_index) {
    const vk::MemoryPropertyFlags flags = GetMemoryTypeProperties(physical_device, memory_type_index);
    return (flags & vk::MemoryPropertyFlagBits::eDeviceLocal) != vk::MemoryPropertyFlags{};
}

bool MemoryUtils::IsMemoryTypeHostCoherent(vk::PhysicalDevice physical_device, std::uint32_t memory_type_index) {
    const vk::MemoryPropertyFlags flags = GetMemoryTypeProperties(physical_device, memory_type_index);
    return (flags & vk::MemoryPropertyFlagBits::eHostCoherent) != vk::MemoryPropertyFlags{};
}

uint32_t MemoryUtils::GetOptimalBufferMemoryType(vk::PhysicalDevice physical_device,
                                                 vk::Device device,
                                                 vk::Buffer buffer,
                                                 vk::MemoryPropertyFlags preferred_properties,
                                                 vk::MemoryPropertyFlags required_properties) {
    const vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(buffer);

    if (auto index = TryFindMemoryType(physical_device,
                                       requirements.memoryTypeBits,
                                       required_properties,
                                       preferred_properties)) {
        return *index;
    }

    throw std::runtime_error("Failed to find suitable memory type for buffer");
}

uint32_t MemoryUtils::GetOptimalImageMemoryType(vk::PhysicalDevice physical_device,
                                                vk::Device device,
                                                vk::Image image,
                                                vk::MemoryPropertyFlags preferred_properties,
                                                vk::MemoryPropertyFlags required_properties) {
    const vk::MemoryRequirements requirements = device.getImageMemoryRequirements(image);

    if (auto index = TryFindMemoryType(physical_device,
                                       requirements.memoryTypeBits,
                                       required_properties,
                                       preferred_properties)) {
        return *index;
    }

    throw std::runtime_error("Failed to find suitable memory type for image");
}

vk::DeviceSize MemoryUtils::CalculateAlignedBufferSize(vk::DeviceSize size, vk::DeviceSize min_alignment) {
    return AlignedSize(size, min_alignment);
}

} // namespace VulkanBackend::Vulkan
