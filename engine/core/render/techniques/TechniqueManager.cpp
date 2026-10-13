module;


module VulkanEngine.TechniqueManager;

import std;
import std.compat;

import vulkan;

import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

namespace VulkanEngine::TechniqueManager {

TechniqueManager::~TechniqueManager() {
    Shutdown();
}

BaseTechnique* TechniqueManager::GetTechnique(uint16_t technique_id) {
    if (technique_id >= techniques_.size()) return nullptr;
    return techniques_[technique_id].base_technique.get();
}

BaseTechnique* TechniqueManager::GetTechnique(TechniqueId id) {
    return GetTechnique(id.value);
}

void TechniqueManager::Shutdown() {
    techniques_.clear();
    type_to_id_.clear();
    intern_to_group_.clear();
    group_technique_.clear();
    group_variant_.clear();
    group_render_state_.clear();
    next_draw_group_ = 0;
}

void TechniqueManager::PollShaders(ShaderSystem::ShaderManager& shaders,
                                   ShaderSystem::PipelineFactory& factory,
                                   std::uint32_t frame_index) {
    for (auto& technique : techniques_) {
        if (technique.base_technique) {
            technique.base_technique->PollAndRebuild(shaders, factory, frame_index);
        }
    }
}

bool TechniqueManager::RebuildForDrawMode(ShaderSystem::ShaderManager& shaders,
                                          ShaderSystem::PipelineFactory& factory,
                                          std::uint32_t draw_mode,
                                          std::uint32_t frame_index) {
    bool ok = true;
    for (auto& technique : techniques_) {
        if (technique.base_technique) {
            ok = technique.base_technique->RebuildForDrawMode(
                     shaders, factory, draw_mode, frame_index) && ok;
        }
    }
    return ok;
}

}

