module;

#include <logging/logging_macros.hpp>

module VulkanEngine.ShaderManager;

import std;
import std.compat;

import logiface;
import vulkan;

import VulkanShared.FileIO;
import VulkanShared.ThreadPool;
import VulkanShared.UserPaths;
import VulkanEngine.CompilerEngine;
import VulkanEngine.ShaderLoader;

namespace VulkanEngine::ShaderSystem {

namespace {
    std::string uuidToHex(const std::array<uint8_t, 16>& uuid) {
        std::string result;
        result.reserve(32);
        for (auto byte : uuid) {
            result += "0123456789abcdef"[byte >> 4];
            result += "0123456789abcdef"[byte & 0xf];
        }
        return result;
    }
}

ShaderManager::ShaderManager(const vk::raii::Device& device, const VulkanBackend::Vulkan::VulkanCapabilities& caps,
                               std::string_view cache_dir)
    : device_(device)
{
    auto uuid_hex = uuidToHex(caps.GetProperties().pipelineCacheUUID);
    // path::operator/ rather than a formatted "dir/file" string: it is the only
    // form that survives a trailing separator or a separator that differs from
    // the platform's.
    cache_path_ = std::filesystem::path{cache_dir} /
                  std::format("pipeline_cache_{}.bin", uuid_hex);

    auto initial_data = VulkanShared::FileIO::ReadBinary(cache_path_);
    vk::PipelineCacheCreateInfo ci{};
    if (!initial_data.empty()) {
        ci.initialDataSize = initial_data.size();
        ci.pInitialData = initial_data.data();
    }

    try {
        cache_ = device_.createPipelineCache(ci);
    } catch (const vk::SystemError&) {
        if (!initial_data.empty()) {
            LOGIFACE_LOG(warn, "Pipeline cache load failed, starting fresh");
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            cache_ = device_.createPipelineCache(ci);
        }
    }
    LOGIFACE_LOG(info, std::format("ShaderManager: pipeline cache at {}",
                                   VulkanShared::UserPaths::ToUtf8(cache_path_)));
}

ShaderManager::~ShaderManager() {
    SerializeCache();
    cache_ = vk::raii::PipelineCache(nullptr);
}

ShaderId ShaderManager::Register(std::string spv_path, std::string slang_path,
                                   ShaderStage stage, std::span<const Binding> bindings,
                                   std::uint64_t binding_hash, std::string entry_point) {
    std::unique_lock lock(slots_mutex_);
    auto slot = std::make_unique<ShaderModuleSlot>();
    slot->spv_path = std::move(spv_path);
    slot->slang_path = std::move(slang_path);
    slot->entry_point = std::move(entry_point);
    slot->stage = stage;
    slot->bindings.assign(bindings.begin(), bindings.end());
    slot->binding_hash = binding_hash;
    slot->manual = false;
    ShaderId id = static_cast<ShaderId>(slots_.size());
    slots_.push_back(std::move(slot));
    return id;
}

ShaderId ShaderManager::RegisterManual(std::string spv_path, std::string slang_path,
                                         ShaderStage stage, std::string entry_point) {
    std::unique_lock lock(slots_mutex_);
    auto slot = std::make_unique<ShaderModuleSlot>();
    slot->spv_path = std::move(spv_path);
    slot->slang_path = std::move(slang_path);
    slot->entry_point = std::move(entry_point);
    slot->stage = stage;
    slot->binding_hash = 0;
    slot->manual = true;
    ShaderId id = static_cast<ShaderId>(slots_.size());
    slots_.push_back(std::move(slot));
    return id;
}

std::expected<vk::ShaderModule, std::string> ShaderManager::GetModule(ShaderId id) {
    std::shared_lock lock(slots_mutex_);
    auto& slot = *slots_[id];
    if (!slot.loaded) {
        lock.unlock();
        if (!LoadSpirvFromDisk(slot)) {
            return std::unexpected("Failed to load shader: " + slot.spv_path);
        }
        lock.lock();
    }
    return *slot.module;
}

const ShaderModuleSlot& ShaderManager::GetSlot(ShaderId id) const {
    std::shared_lock lock(slots_mutex_);
    return *slots_[id];
}

ShaderId ShaderManager::FindBySlangPath(std::string_view path) const {
    std::shared_lock lock(slots_mutex_);
    for (ShaderId i = 0; i < static_cast<ShaderId>(slots_.size()); ++i) {
        if (slots_[i]->slang_path == path) return i;
    }
    return static_cast<ShaderId>(-1);
}

ShaderId ShaderManager::FindBySlangFilename(std::string_view filename) const {
    std::shared_lock lock(slots_mutex_);
    for (ShaderId i = 0; i < static_cast<ShaderId>(slots_.size()); ++i) {
        if (std::filesystem::path(slots_[i]->slang_path).filename() == filename) return i;
    }
    return static_cast<ShaderId>(-1);
}

std::vector<ShaderId> ShaderManager::FindAllBySlangPath(std::string_view path) const {
    std::shared_lock lock(slots_mutex_);
    std::vector<ShaderId> ids;
    for (ShaderId i = 0; i < static_cast<ShaderId>(slots_.size()); ++i) {
        if (slots_[i]->slang_path == path) ids.push_back(i);
    }
    return ids;
}

std::vector<ShaderId> ShaderManager::FindAllBySlangFilename(std::string_view filename) const {
    std::shared_lock lock(slots_mutex_);
    std::vector<ShaderId> ids;
    for (ShaderId i = 0; i < static_cast<ShaderId>(slots_.size()); ++i) {
        if (std::filesystem::path(slots_[i]->slang_path).filename() == filename) ids.push_back(i);
    }
    return ids;
}

std::vector<std::string> ShaderManager::GetSlangDirectories() const {
    std::shared_lock lock(slots_mutex_);
    std::vector<std::string> dirs;
    for (const auto& slot : slots_) {
        auto dir = std::filesystem::weakly_canonical(
            std::filesystem::path(slot->slang_path).parent_path()).string();
        if (std::ranges::find(dirs, dir) == dirs.end()) dirs.push_back(std::move(dir));
    }
    return dirs;
}

std::uint64_t ShaderManager::GetVersion(ShaderId id) const {
    std::shared_lock lock(slots_mutex_);
    return slots_[id]->version.load(std::memory_order_acquire);
}

std::future<bool> ShaderManager::RequestReload(ShaderId id) {
    return VulkanShared::ThreadPool::Global().Enqueue([this, id]() -> bool {
        ShaderModuleSlot* slot = nullptr;
        {
            std::shared_lock lock(slots_mutex_);
            slot = slots_[id].get();
        }

        std::vector<std::uint32_t> spirv;
#if VKENGINE_HOT_RELOAD
        VulkanEngine::ShaderSystem::CompileRequest req{};
        req.source_path = slot->slang_path;
        // The slot's own entry point: a variant row recompiles its wrapper, not
        // "main" (which would clobber the variant's module).
        req.entry_point = slot->entry_point;
        req.stage = slot->stage;
        req.optimization_level = 0;

        auto result = CompilerEngine::Compile(req);
        if (!result.has_value()) {
            LOGIFACE_LOG(error, std::format("ShaderManager: compilation failed: {}", result.error()));
            return false;
        }

        if (!slot->manual && result->binding_hash != slot->binding_hash) {
            LOGIFACE_LOG(error, std::format("ShaderManager: {} — bindings changed, C++ rebuild required", slot->slang_path));
            return false;
        }

        spirv = std::move(result->spirv);
        LOGIFACE_LOG(debug, std::format("ShaderManager: recompiled {} ({} bytes, stage {})",
                                        slot->slang_path, spirv.size() * sizeof(std::uint32_t),
                                        static_cast<std::uint32_t>(slot->stage)));
        VulkanShared::FileIO::WriteBinary(slot->spv_path,
            std::span{reinterpret_cast<const std::byte*>(spirv.data()),
                      spirv.size() * sizeof(std::uint32_t)});
#else
        auto spirv_result = VulkanEngine::ShaderLoader::ShaderLoader::LoadSpirv(slot->spv_path);
        if (!spirv_result) {
            LOGIFACE_LOG(error, std::format("ShaderManager: reload failed for {}: {}", slot->spv_path, spirv_result.error()));
            return false;
        }
        spirv = std::move(*spirv_result);
#endif

        // Create the module off-lock, then publish it atomically with the version
        // bump. Readers (GetModule/GetVersion/GetSlot) take the shared lock, and
        // runtime pipeline creation is gated on GetVersion() changing, so the
        // retired module has no live consumers when it is destroyed here.
        vk::ShaderModuleCreateInfo info({},
            spirv.size() * sizeof(std::uint32_t), spirv.data());
        vk::raii::ShaderModule new_module(device_, info);
        {
            std::unique_lock lock(slots_mutex_);
            slot->module = std::move(new_module);
            slot->loaded = true;
            slot->version.fetch_add(1, std::memory_order_release);
        }
        LOGIFACE_LOG(debug, std::format("ShaderManager: hot-reloaded {} -> v{}",
                                        slot->slang_path,
                                        slot->version.load(std::memory_order_acquire)));
        return true;
    });
}

std::future<bool> ShaderManager::RequestReloadByPath(std::string_view slang_path) {
    const auto ids = FindAllBySlangPath(slang_path);
    if (ids.empty()) {
        std::promise<bool> p;
        p.set_value(false);
        return p.get_future();
    }
    if (ids.size() == 1) {
        return RequestReload(ids.front());
    }
    // Several variants share this source: reload them all and aggregate.
    std::vector<std::future<bool>> futures;
    futures.reserve(ids.size());
    for (const ShaderId id : ids) {
        futures.push_back(RequestReload(id));
    }
    return VulkanShared::ThreadPool::Global().Enqueue(
        [futures = std::move(futures)]() mutable -> bool {
            bool ok = true;
            for (auto& f : futures) {
                ok = f.get() && ok;
            }
            return ok;
        });
}

vk::PipelineCache ShaderManager::GetPipelineCache() const {
    return *cache_;
}

const vk::raii::PipelineCache& ShaderManager::GetPipelineCacheRAII() const {
    return cache_;
}

void ShaderManager::SerializeCache() {
    try {
        auto data = cache_.getData();
        if (!data.empty()) {
            VulkanShared::FileIO::AtomicWrite(cache_path_,
                std::span{reinterpret_cast<const std::byte*>(data.data()), data.size()});
        }
    } catch (const std::exception& e) {
        LOGIFACE_LOG(error, std::format("ShaderManager: failed to serialize pipeline cache: {}", e.what()));
    }
}

void ShaderManager::FlushCache() {
    SerializeCache();
}

bool ShaderManager::LoadSpirvFromDisk(ShaderModuleSlot& slot) {
    LOGIFACE_LOG(debug, std::format("ShaderManager: loading SPIR-V module from {}", slot.spv_path));
    auto result = VulkanEngine::ShaderLoader::ShaderLoader::LoadSpirv(slot.spv_path);
    if (!result) {
        LOGIFACE_LOG(error, std::format("ShaderManager: {} — {}", slot.spv_path, result.error()));
        return false;
    }
    auto& spirv = *result;
    if (spirv.empty()) {
        LOGIFACE_LOG(error, std::format("ShaderManager: empty SPIR-V for {}", slot.spv_path));
        return false;
    }
    vk::ShaderModuleCreateInfo info({},
        spirv.size() * sizeof(std::uint32_t), spirv.data());
    slot.module = vk::raii::ShaderModule(device_, info);
    slot.loaded = true;
    return true;
}

} // namespace VulkanEngine::ShaderSystem
