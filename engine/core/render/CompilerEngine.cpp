module;

#if VKENGINE_HOT_RELOAD
#include <slang.h>
#endif

#include <logging/logging_macros.hpp>

module VulkanEngine.CompilerEngine;

import std;
import std.compat;
import ShaderReflection;

import logiface;

namespace VulkanEngine::ShaderSystem {

#if VKENGINE_HOT_RELOAD

namespace {

    std::string blobToString(slang::IBlob* blob) {
        if (!blob) return "";
        return std::string(static_cast<const char*>(blob->getBufferPointer()),
                           blob->getBufferSize());
    }

    std::string sourceContentHash(std::string_view s) {
        std::uint64_t h = 0xcbf29ce484222325ULL;
        for (char c : s) h = (h ^ static_cast<std::uint8_t>(c)) * 0x100000001b3ULL;
        std::string out;
        out.reserve(16);
        constexpr std::string_view hex = "0123456789abcdef";
        for (int i = 7; i >= 0; --i) {
            out += hex[(h >> (i * 8 + 4)) & 0xf];
            out += hex[(h >> (i * 8)) & 0xf];
        }
        return out;
    }

    SlangStage slangStageFromShaderStage(ShaderStage s) {
        switch (s) {
        case ShaderStage::eVertex:         return SLANG_STAGE_VERTEX;
        case ShaderStage::eFragment:       return SLANG_STAGE_FRAGMENT;
        case ShaderStage::eCompute:        return SLANG_STAGE_COMPUTE;
        case ShaderStage::eRayGeneration:  return SLANG_STAGE_RAY_GENERATION;
        case ShaderStage::eIntersection:   return SLANG_STAGE_INTERSECTION;
        case ShaderStage::eAnyHit:         return SLANG_STAGE_ANY_HIT;
        case ShaderStage::eClosestHit:     return SLANG_STAGE_CLOSEST_HIT;
        case ShaderStage::eMiss:           return SLANG_STAGE_MISS;
        case ShaderStage::eCallable:       return SLANG_STAGE_CALLABLE;
        case ShaderStage::eMesh:           return SLANG_STAGE_MESH;
        case ShaderStage::eAmplification:  return SLANG_STAGE_AMPLIFICATION;
        case ShaderStage::eHull:           return SLANG_STAGE_HULL;
        case ShaderStage::eDomain:         return SLANG_STAGE_DOMAIN;
        case ShaderStage::eGeometry:       return SLANG_STAGE_GEOMETRY;
        }
        return SLANG_STAGE_FRAGMENT;
    }

    std::string_view bindingTypeEnumSlang(std::string_view slangType) {
        if (slangType == "ConstantBuffer")           return "eUniformBuffer";
        if (slangType == "ParameterBlock")           return "eParameterBlock";
        if (slangType == "TextureBuffer")            return "eTextureBuffer";
        if (slangType == "ShaderStorageBuffer")      return "eStorageBuffer";
        if (slangType == "StructuredBuffer")         return "eStructuredBuffer";
        if (slangType == "ByteAddressBuffer")        return "eByteAddressBuffer";
        if (slangType == "Texture1D")                return "eTexture1D";
        if (slangType == "Texture2D")                return "eTexture2D";
        if (slangType == "Texture3D")                return "eTexture3D";
        if (slangType == "TextureCube")              return "eTextureCube";
        if (slangType == "SamplerState")             return "eSampler";
        if (slangType == "AccelerationStructure")    return "eAccelerationStructure";
        if (slangType == "SubpassInput")             return "eSubpassInput";
        return "eResource";
    }

    BindingType bindingTypeFromString(std::string_view s) {
        if (s == "eUniformBuffer")        return BindingType::eUniformBuffer;
        if (s == "eParameterBlock")       return BindingType::eParameterBlock;
        if (s == "eTextureBuffer")        return BindingType::eTextureBuffer;
        if (s == "eStorageBuffer")        return BindingType::eStorageBuffer;
        if (s == "eStructuredBuffer")     return BindingType::eStructuredBuffer;
        if (s == "eByteAddressBuffer")    return BindingType::eByteAddressBuffer;
        if (s == "eTexture1D")            return BindingType::eTexture1D;
        if (s == "eTexture2D")            return BindingType::eTexture2D;
        if (s == "eTexture3D")            return BindingType::eTexture3D;
        if (s == "eTextureCube")          return BindingType::eTextureCube;
        if (s == "eSampler")              return BindingType::eSampler;
        if (s == "eAccelerationStructure") return BindingType::eAccelerationStructure;
        if (s == "eSubpassInput")         return BindingType::eSubpassInput;
        if (s == "eTextureBuffer")        return BindingType::eTextureBuffer;
        return BindingType::eResource;
    }

    struct BindingEntry {
        std::string type_enum;
        uint32_t set;
        uint32_t binding;
        uint32_t count;
    };

    std::string typeName(slang::TypeLayoutReflection* tl) {
        auto kind = tl->getKind();
        switch (kind) {
        case slang::TypeReflection::Kind::ConstantBuffer:   return "ConstantBuffer";
        case slang::TypeReflection::Kind::ParameterBlock:    return "ParameterBlock";
        case slang::TypeReflection::Kind::TextureBuffer:     return "TextureBuffer";
        case slang::TypeReflection::Kind::ShaderStorageBuffer: return "ShaderStorageBuffer";
        case slang::TypeReflection::Kind::Resource: {
            auto shape = tl->getResourceShape();
            auto base = shape & SLANG_RESOURCE_BASE_SHAPE_MASK;
            if (base == SLANG_STRUCTURED_BUFFER) return "StructuredBuffer";
            if (base == SLANG_BYTE_ADDRESS_BUFFER) return "ByteAddressBuffer";
            if (base == SLANG_TEXTURE_1D) return "Texture1D";
            if (base == SLANG_TEXTURE_2D) return "Texture2D";
            if (base == SLANG_TEXTURE_3D) return "Texture3D";
            if (base == SLANG_TEXTURE_CUBE) return "TextureCube";
            if (base == SLANG_ACCELERATION_STRUCTURE) return "AccelerationStructure";
            if (base == SLANG_TEXTURE_SUBPASS) return "SubpassInput";
            return "Resource";
        }
        case slang::TypeReflection::Kind::SamplerState:     return "SamplerState";
        case slang::TypeReflection::Kind::Array: {
            auto etl = tl->getElementTypeLayout();
            return typeName(etl) + "[]";
        }
        case slang::TypeReflection::Kind::Struct: {
            auto t = tl->getType();
            return t ? t->getName() : "struct";
        }
        case slang::TypeReflection::Kind::None:
            return "void";
        default:
            return tl->getName() ? tl->getName() : "unknown";
        }
    }

    slang::TypeReflection::Kind varTypeKind(slang::VariableLayoutReflection* var) {
        auto type = var->getType();
        if (type) return type->getKind();
        auto tl = var->getTypeLayout();
        return tl ? tl->getKind() : slang::TypeReflection::Kind::None;
    }

    bool collectBinding(slang::VariableLayoutReflection* var, BindingEntry& out) {
        auto tl = var->getTypeLayout();
        if (!tl) return false;

        bool found = false;
        int catCount = var->getCategoryCount();
        for (int ci = 0; ci < catCount; ++ci) {
            auto cat = var->getCategoryByIndex(ci);
            if (cat == slang::ParameterCategory::DescriptorTableSlot) {
                out.set = static_cast<uint32_t>(var->getBindingSpace(cat));
                out.binding = static_cast<uint32_t>(var->getOffset(cat));
                found = true;
                break;
            }
        }

        if (!found) return false;

        auto rtype = typeName(tl);
        out.type_enum = bindingTypeEnumSlang(rtype).data();
        out.count = 1;
        return true;
    }

    bool isUnwrappedParameterBlock(slang::VariableLayoutReflection* var) {
        auto vk = varTypeKind(var);
        if (vk == slang::TypeReflection::Kind::ParameterBlock) return true;
        if (vk != slang::TypeReflection::Kind::Struct) return false;
        int catCount = var->getCategoryCount();
        for (int ci = 0; ci < catCount; ++ci) {
            if (var->getCategoryByIndex(ci) == slang::ParameterCategory::DescriptorTableSlot)
                return true;
        }
        return false;
    }

    void collectAllBindingsRecursive(slang::VariableLayoutReflection* var,
                                      std::vector<BindingEntry>& entries) {
        auto tl = var->getTypeLayout();
        if (!tl) return;
        auto varKind = varTypeKind(var);

        if (varKind == slang::TypeReflection::Kind::ConstantBuffer) {
            BindingEntry entry;
            if (collectBinding(var, entry))
                entries.push_back(std::move(entry));
            return;
        }

        if (isUnwrappedParameterBlock(var)) {
            BindingEntry entry;
            if (collectBinding(var, entry)) {
                entry.type_enum = "eParameterBlock";
                entries.push_back(std::move(entry));
            }
            // 2026.18 added getContentVarLayout: the block's content variable
            // in one hop (element layout for containers, null for
            // non-container types - no hand-rolled unwrap needed).
            if (auto* content = tl->getContentVarLayout(); content != nullptr) {
                collectAllBindingsRecursive(content, entries);
            }
            return;
        }

        BindingEntry entry;
        if (collectBinding(var, entry))
            entries.push_back(std::move(entry));

        auto kind = tl->getKind();
        if (kind == slang::TypeReflection::Kind::Struct) {
            int fc = tl->getFieldCount();
            for (int i = 0; i < fc; ++i)
                collectAllBindingsRecursive(tl->getFieldByIndex(i), entries);
        }
    }

    std::uint64_t bindingHashForEntries(const std::vector<BindingEntry>& entries,
                                          std::string_view stage_enum_str) {
        auto fnv1a_64 = [](std::string_view s, std::uint64_t h = 0xcbf29ce484222325ULL) -> std::uint64_t {
            for (char c : s) h = (h ^ static_cast<std::uint8_t>(c)) * 0x100000001b3ULL;
            return h;
        };

        std::string concat;
        for (const auto& e : entries) {
            concat += e.type_enum;
            concat += ',';
            concat += std::to_string(e.set);
            concat += ',';
            concat += std::to_string(e.binding);
            concat += ',';
            concat += std::to_string(e.count);
            concat += ';';
        }

        std::uint64_t stage_val = 0;
        if (stage_enum_str == "eVertex") stage_val = 0;
        else if (stage_enum_str == "eFragment") stage_val = 1;
        else if (stage_enum_str == "eCompute") stage_val = 2;
        else if (stage_enum_str == "eRayGeneration") stage_val = 3;
        else if (stage_enum_str == "eIntersection") stage_val = 4;
        else if (stage_enum_str == "eAnyHit") stage_val = 5;
        else if (stage_enum_str == "eClosestHit") stage_val = 6;
        else if (stage_enum_str == "eMiss") stage_val = 7;
        else if (stage_enum_str == "eCallable") stage_val = 8;
        else if (stage_enum_str == "eMesh") stage_val = 9;
        else if (stage_enum_str == "eAmplification") stage_val = 10;
        else if (stage_enum_str == "eHull") stage_val = 11;
        else if (stage_enum_str == "eDomain") stage_val = 12;
        else if (stage_enum_str == "eGeometry") stage_val = 13;

        return (stage_val << 56) | fnv1a_64(concat);
    }

    std::string_view stageEnumToString(SlangStage s) {
        switch (s) {
        case SLANG_STAGE_VERTEX:         return "eVertex";
        case SLANG_STAGE_FRAGMENT:       return "eFragment";
        case SLANG_STAGE_COMPUTE:        return "eCompute";
        case SLANG_STAGE_RAY_GENERATION: return "eRayGeneration";
        case SLANG_STAGE_INTERSECTION:   return "eIntersection";
        case SLANG_STAGE_ANY_HIT:        return "eAnyHit";
        case SLANG_STAGE_CLOSEST_HIT:    return "eClosestHit";
        case SLANG_STAGE_MISS:           return "eMiss";
        case SLANG_STAGE_CALLABLE:       return "eCallable";
        case SLANG_STAGE_MESH:           return "eMesh";
        case SLANG_STAGE_AMPLIFICATION:  return "eAmplification";
        case SLANG_STAGE_HULL:           return "eHull";
        case SLANG_STAGE_DOMAIN:         return "eDomain";
        case SLANG_STAGE_GEOMETRY:       return "eGeometry";
        default: return "eFragment";
        }
    }

    // Long-lived Slang global session for the engine / hot-reload path. The
    // deprecated compile-request API (spCreateCompileRequest / spCompile, the
    // same path `slangc` uses) is used here deliberately: the modern Session
    // API (`loadModuleFromSourceString` → … → `getLayout`) corrupts the heap at
    // session teardown on Linux (see
    // docs/Slang-session-teardown-heap-corruption-bug-and-workaround.md), while
    // the old API is unaffected. The global session is created once and never
    // released — it is meant to live for the application's lifetime.
    struct SlangRuntime {
        slang::IGlobalSession* global = nullptr;
        SlangProfileID profile = SLANG_PROFILE_UNKNOWN;

        SlangRuntime() {
            if (SLANG_FAILED(slang::createGlobalSession(&global))) {
                global = nullptr;
                return;
            }
            profile = global->findProfile("SPIRV_1_6");
        }
    };

    SlangRuntime& GetSlangRuntime() {
        static SlangRuntime runtime; // thread-safe one-time init (magic static)
        return runtime;
    }

    std::mutex& CompileMutex() {
        static std::mutex m;
        return m;
    }

} // anonymous namespace

std::expected<CompileResult, std::string>
CompilerEngine::Compile(const CompileRequest& req) {
    auto& rt = GetSlangRuntime();
    if (!rt.global)
        return std::unexpected("Slang runtime not initialized");

    // The compile-request API is not documented thread-safe; serialise
    // compiles. Reloads are rare (user edits), so contention is nil.
    std::lock_guard lock(CompileMutex());

    std::string source = req.source_text;
    if (source.empty()) {
        std::ifstream f(req.source_path, std::ios::binary | std::ios::ate);
        if (!f) {
            return std::unexpected("cannot open source file: " + req.source_path);
        }
        f.seekg(0);
        source.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // Every compile request is a fresh, independent compile; the translation
    // unit name only shows up in diagnostics. Keep it unique per source
    // version anyway so a stale module cache can never shadow newer content.
    auto moduleName = req.source_path + "_" + sourceContentHash(source);

    SlangCompileRequest* request = spCreateCompileRequest(rt.global);
    if (!request)
        return std::unexpected("createCompileRequest failed");

    SlangOptimizationLevel optLevel = SLANG_OPTIMIZATION_LEVEL_NONE;
    switch (req.optimization_level) {
        case 0: break;
        case 1: optLevel = SLANG_OPTIMIZATION_LEVEL_DEFAULT; break;
        case 2: optLevel = SLANG_OPTIMIZATION_LEVEL_HIGH; break;
        case 3: optLevel = SLANG_OPTIMIZATION_LEVEL_MAXIMAL; break;
    }

    spSetCodeGenTarget(request, SLANG_SPIRV);
    spSetTargetProfile(request, 0, rt.profile);
    spSetTargetFlags(request, 0, SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY);
    spSetOptimizationLevel(request, optLevel);

    auto slangStage = slangStageFromShaderStage(req.stage);
    const int tuIndex = spAddTranslationUnit(
        request, SLANG_SOURCE_LANGUAGE_SLANG, moduleName.c_str());
    spAddTranslationUnitSourceString(
        request, tuIndex, req.source_path.c_str(), source.c_str());
    spAddEntryPoint(request, tuIndex, req.entry_point.c_str(), slangStage);

    SlangResult compileResult = spCompile(request);
    if (SLANG_FAILED(compileResult)) {
        std::string msg;
        ISlangBlob* diagBlob = nullptr;
        if (SLANG_SUCCEEDED(spGetDiagnosticOutputBlob(request, &diagBlob))) {
            msg = blobToString(diagBlob);
            diagBlob->release();
        }
        if (msg.empty()) {
            if (const char* raw = spGetDiagnosticOutput(request)) msg = raw;
        }
        spDestroyCompileRequest(request);
        return std::unexpected("compile failed: " + msg);
    }

    size_t codeSize = 0;
    const void* code = spGetEntryPointCode(request, 0, &codeSize);
    if (!code || codeSize == 0) {
        spDestroyCompileRequest(request);
        return std::unexpected("getEntryPointCode failed");
    }

    auto* layout = slang::ProgramLayout::get(request);
    if (!layout) {
        spDestroyCompileRequest(request);
        return std::unexpected("getReflection failed");
    }

    auto* entryPointLayout = layout->getEntryPointByIndex(0);
    auto actualStage = entryPointLayout ? entryPointLayout->getStage() : slangStage;

    std::vector<BindingEntry> entries;
    int paramCount = layout->getParameterCount();
    for (int i = 0; i < paramCount; ++i) {
        auto* var = layout->getParameterByIndex(i);
        collectAllBindingsRecursive(var, entries);
    }

    CompileResult result;
    auto* spv_data = static_cast<const uint32_t*>(code);
    auto spv_size = codeSize / sizeof(uint32_t);
    result.spirv.assign(spv_data, spv_data + spv_size);

    auto stageStr = stageEnumToString(actualStage);
    result.binding_hash = bindingHashForEntries(entries, stageStr);

    result.bindings.reserve(entries.size());
    for (const auto& e : entries) {
        result.bindings.push_back(Binding{
            bindingTypeFromString(e.type_enum),
            e.set, e.binding, e.count
        });
    }

    if (const char* rawDiag = spGetDiagnosticOutput(request); rawDiag && *rawDiag)
        result.diagnostics = rawDiag;

    spDestroyCompileRequest(request);
    return result;
}

#else

std::expected<CompileResult, std::string>
CompilerEngine::Compile(const CompileRequest& /*req*/) {
    return std::unexpected("hot-reload disabled at build time");
}

#endif

} // namespace VulkanEngine::ShaderSystem
