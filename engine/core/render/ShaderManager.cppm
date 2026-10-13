module;

export module VulkanEngine.ShaderManager;

import std;
import vulkan;
import ShaderReflection;
import VulkanBackend.Vulkan.VulkanCapabilities;

export namespace VulkanEngine::ShaderSystem {

using ShaderId = std::uint16_t;

struct ShaderModuleSlot {
    std::string spv_path;
    std::string slang_path;
    // Named entry point in the module. One source may be compiled into several
    // entry-point wrappers, one .spv row per wrapper; each row keeps its own
    // entry point so hot reload recompiles the right one (never "main").
    std::string entry_point{"main"};
    ShaderStage stage;
    std::vector<Binding> bindings;
    std::uint64_t binding_hash;
    vk::raii::ShaderModule module{nullptr};
    std::atomic<std::uint64_t> version{0};
    bool loaded{false};
    bool manual{false};
};

class ShaderManager {
public:
    ShaderManager(const vk::raii::Device& device, const VulkanBackend::Vulkan::VulkanCapabilities& caps,
                  std::string_view cache_dir);
    ~ShaderManager();

    ShaderManager(const ShaderManager&) = delete;
    ShaderManager& operator=(const ShaderManager&) = delete;
    ShaderManager(ShaderManager&&) = delete;
    ShaderManager& operator=(ShaderManager&&) = delete;

    ShaderId Register(std::string spv_path, std::string slang_path,
                      ShaderStage stage, std::span<const Binding> bindings,
                      std::uint64_t binding_hash, std::string entry_point = "main");

    ShaderId RegisterManual(std::string spv_path, std::string slang_path,
                            ShaderStage stage, std::string entry_point = "main");

    [[nodiscard]] std::expected<vk::ShaderModule, std::string> GetModule(ShaderId id);
    [[nodiscard]] const ShaderModuleSlot& GetSlot(ShaderId id) const;
    // First slot with this source path / filename; returns -1 when none.
    [[nodiscard]] ShaderId FindBySlangPath(std::string_view path) const;
    [[nodiscard]] ShaderId FindBySlangFilename(std::string_view filename) const;
    // Every slot registered from this source path / filename. One source may
    // back several entry-point modules (variants), so a reload must enumerate
    // and recompile all of them rather than the first.
    [[nodiscard]] std::vector<ShaderId> FindAllBySlangPath(std::string_view path) const;
    [[nodiscard]] std::vector<ShaderId> FindAllBySlangFilename(std::string_view filename) const;
    [[nodiscard]] std::vector<std::string> GetSlangDirectories() const;
    [[nodiscard]] std::uint64_t GetVersion(ShaderId id) const;

    std::future<bool> RequestReload(ShaderId id);
    // Reloads every slot registered from `slang_path` (all variants of one
    // source), resolving to false if none. One save must not clobber other
    // entry points of the same source with "main".
    std::future<bool> RequestReloadByPath(std::string_view slang_path);

    [[nodiscard]] vk::PipelineCache GetPipelineCache() const;
    [[nodiscard]] const vk::raii::PipelineCache& GetPipelineCacheRAII() const;

    void SerializeCache();
    void FlushCache();

private:
    bool LoadSpirvFromDisk(ShaderModuleSlot& slot);

    const vk::raii::Device& device_;
    vk::raii::PipelineCache cache_{nullptr};
    std::vector<std::unique_ptr<ShaderModuleSlot>> slots_;
    std::filesystem::path cache_path_;
    mutable std::shared_mutex slots_mutex_;
};

} // namespace VulkanEngine::ShaderSystem
