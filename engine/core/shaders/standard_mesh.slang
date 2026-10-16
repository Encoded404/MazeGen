[[vk::binding(0, 0)]]
Sampler2D<float4> allTextures[];

// Camera position (fragment-stage push constant, 16 bytes at offset 0)
[[vk::push_constant]]
float4 camera_position;

// Scene header (set 4, binding 0) — C layout matches the C++ mirror in
// SceneRenderer.cppm (SceneHeader: 3 x float4 + uint = 52 B).
struct SceneHeader {
    float4 ambient_color;
    float4 sun_direction;
    float4 sun_color;
    uint light_count;
};

[[vk::binding(0, 4)]]
StructuredBuffer<SceneHeader, CDataLayout> sceneHeader;

// Light array (set 4, binding 1) — BlockArray-backed unbounded array.
// Type dispatch lives in position.w (see the CPU mirror in SceneRenderer.cppm
// for the authoritative contract comment):
//   position: xyz = world position of a point/spot light; w = light type
//             (0 = directional, 1 = point, 2 = spot).
//   direction: xyz = -L for directional lights; the spot cone axis pointing
//             AWAY from the light (unused otherwise).
//   color:    rgb tint, w = intensity.
//   params:   x = range (point/spot; <= 0 = infinite), y/z = cos(inner)/
//             cos(outer) spot cone angles (unused otherwise).
struct Light {
    float4 position;
    float4 color;
    float4 direction;
    float4 params;
};

[[vk::binding(1, 4)]]
StructuredBuffer<Light, CDataLayout> sceneLights[];

// Per-material data (set 5, binding 0) — C layout matches the C++
// DefaultMeshPerMaterialData exactly (64 B). Change both together.
struct MaterialData {
    uint albedo_texture;
    uint normal_texture;
    uint orm_texture;
    uint emissive_texture;
    uint flags;          // bits0..1 normal encoding; bit2 alpha_mask; bit3 has_transform
    uint uv_sets;
    float roughness_factor;
    float metallic_factor;
    float ao_factor;
    float normal_scale;
    float occlusion_strength;
    float alpha_cutoff;
    float4 emissive_factor; // rgb + strength
};

// Declared TextureNormalEncoding (content, not derived from the upload).
static const uint kNormalEncodingStandard = 0u;
static const uint kNormalEncodingFull = 1u;
static const uint kNormalEncodingBent = 2u;

[[vk::binding(0, 5)]]
StructuredBuffer<MaterialData, CDataLayout> materialBuffer[];

struct PSInput {
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float2 texCoord : TEXCOORD1;
    [[vk::location(2)]] uint materialId : TEXCOORD2;
    [[vk::location(3)]] float3 worldPos : TEXCOORD3;
    [[vk::location(4)]] uint ormPacked : TEXCOORD4;   // unorm8: [7:0]=AO, [15:8]=roughness, [23:16]=metallic
    [[vk::location(5)]] float3 tangent : TEXCOORD5;    // world-space tangent
    [[vk::location(6)]] float handedness : TEXCOORD6;  // bitangent handedness, +1/-1
};

// UV1 variant input: the same interpolants plus the out-of-line UV1 at
// location 7. Only the UV1 pipeline declares this.
struct PSInputUV1 {
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float2 texCoord : TEXCOORD1;
    [[vk::location(2)]] uint materialId : TEXCOORD2;
    [[vk::location(3)]] float3 worldPos : TEXCOORD3;
    [[vk::location(4)]] uint ormPacked : TEXCOORD4;
    [[vk::location(5)]] float3 tangent : TEXCOORD5;
    [[vk::location(6)]] float handedness : TEXCOORD6;
    [[vk::location(7)]] float2 uv1 : TEXCOORD7;
};

// Resolves a texture slot's UV from its 2-bit set selector: set 0 uses the
// inline UV0, set 1 uses the out-of-line UV1. `slot` is the selector index
// (0=albedo, 1=normal, 2=orm, 3=emissive).
float2 SlotUV(uint uv_sets, uint slot, float2 uv0, float2 uv1) {
    return ((uv_sets >> (slot * 2u)) & 0x3u) == 1u ? uv1 : uv0;
}

// Cook-Torrance BRDF

static const float PI = 3.14159265359;

float D_GGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float G_SchlickGGX(float NdotV, float roughness) {
    float a = roughness * roughness;
    float k = a * 0.5;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

float3 F_Schlick(float3 F0, float cosTheta) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Shared shading body. `uv0`/`uv1` are the two UV sets; each texture slot
// selects between them through MaterialData.uv_sets.
float4 Shade(PSInput input, float2 uv0, float2 uv1) {
    uint block = input.materialId / 256u;
    uint elem  = input.materialId % 256u;
    MaterialData mat = materialBuffer[NonUniformResourceIndex(block)][elem];

    // Per-slot UV selection (0=albedo, 1=normal, 2=orm, 3=emissive).
    float2 uv_albedo = SlotUV(mat.uv_sets, 0u, uv0, uv1);
    float2 uv_normal = SlotUV(mat.uv_sets, 1u, uv0, uv1);
    float2 uv_orm    = SlotUV(mat.uv_sets, 2u, uv0, uv1);

    // Sample textures. Albedo alpha is carried through to the output (the
    // alpha-blend render state relies on it); opaque textures are at 1.0 so
    // the opaque path is unchanged.
    float4 albedo_sample = allTextures[NonUniformResourceIndex(mat.albedo_texture)]
        .Sample(uv_albedo);
    float3 albedo = albedo_sample.rgb;

    // Effective ORM factor layer, packed unorm8 by the CPU gather pass: the
    // object's OrmOverride if present, else the material's own factors.
    // Multiplied with the material's ORM texture per the glTF convention.
    float3 orm_factors = float3(
        float(input.ormPacked & 0xFFu),
        float((input.ormPacked >> 8) & 0xFFu),
        float((input.ormPacked >> 16) & 0xFFu)) / 255.0;

    float3 orm = orm_factors;
    if (mat.orm_texture != 0) {
        orm *= allTextures[NonUniformResourceIndex(mat.orm_texture)]
            .Sample(uv_orm).rgb;
    }

    float ao = orm.r;
    float roughness = max(orm.g, 0.04);
    float metallic = orm.b;

    // Vectors
    float3 N = normalize(input.normal);
    float3 V = normalize(camera_position.xyz - input.worldPos);

    // Tangent-space normal mapping (glTF convention: tangent xyz + bitangent
    // sign w; bitangent reconstructed, never stored). The TBN frame is built
    // from the interpolated geometric normal so the frame stays orthonormal
    // even where the tangent drifts slightly from perpendicularity.
    //
    // Normal decode follows the DECLARED encoding:
    //   Standard: reconstruct z from RG (BC5 or RGB maps; stored B ignored).
    //   Full/Bent: use RGB as authored, scale xy.
    if (mat.normal_texture != 0) {
        float3 t = normalize(input.tangent);
        float3 b = cross(N, t) * input.handedness;
        float3x3 TBN = float3x3(t, b, N);
        float3 sampled = allTextures[NonUniformResourceIndex(mat.normal_texture)]
            .Sample(uv_normal).xyz;
        float3 n_ts;
        const uint encoding = mat.flags & 0x3u;
        if (encoding == kNormalEncodingStandard) {
            float2 xy = sampled.rg * 2.0 - 1.0;
            xy *= mat.normal_scale;
            float z = sqrt(saturate(1.0 - dot(xy, xy)));
            n_ts = float3(xy, z);
        } else {
            n_ts = sampled * 2.0 - 1.0;
            n_ts.xy *= mat.normal_scale;
        }
        N = normalize(mul(normalize(n_ts), TBN));
    }

    // Cook-Torrance BRDF
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float3 combined_direct = float3(0.0, 0.0, 0.0);
    const uint lc = sceneHeader[0].light_count;

    // Accumulate all lights. The type dispatch reads Light.position.w; the
    // full params contract is documented on the CPU mirror (SceneRenderer.cppm
    // Light struct):
    //   type 0 (directional): L from -direction.xyz, no attenuation.
    //   type 1 (point):       L from position.xyz; inverse-square falloff
    //                         windowed by params.x range (<= 0 = infinite).
    //   type 2 (spot):        as point; direction.xyz is the cone axis
    //                         pointing AWAY from the light, params.y/z =
    //                         cos(inner)/cos(outer). The cone edge fades
    //                         linearly in cosine space between the angles
    //                         (t * t; a deliberate simplification of glTF's
    //                         smoothstep edge).
    for (uint li = 0; li < lc; li++) {
        uint lblock = li / 256u;
        uint lelem  = li % 256u;
        Light lt = sceneLights[NonUniformResourceIndex(lblock)][lelem];

        const uint light_type = uint(lt.position.w);
        float atten = 1.0;
        float3 L;
        if (light_type == 0u) {
            L = normalize(-lt.direction.xyz);
        } else {
            float3 to_light = lt.position.xyz - input.worldPos;
            float d = length(to_light);
            L = to_light / max(d, 1e-4);
            atten = 1.0 / (d * d + 1e-6);
            const float range = lt.params.x;
            if (range > 0.0) {
                const float r = d / range;
                atten *= saturate(1.0 - r * r);
            }
            if (light_type == 2u) {
                const float cos_axis = dot(normalize(-lt.direction.xyz), L);
                const float cos_i = lt.params.y;
                const float cos_o = lt.params.z;
                const float t = saturate((cos_axis - cos_o) / max(cos_i - cos_o, 1e-4));
                atten *= t * t;
            }
        }

        // Fragments facing away from the light contribute nothing. Gate the
        // whole term here instead of letting the 0.001 clamp leak diffuse
        // light onto backfacing geometry.
        if (dot(N, L) <= 0.0) {
            continue;
        }

        float3 H = normalize(V + L);
        float NdotV = max(dot(N, V), 0.001);
        float NdotL = max(dot(N, L), 0.001);
        float NdotH = max(dot(N, H), 0.001);

        float D = D_GGX(NdotH, roughness);
        float G = G_Smith(NdotV, NdotL, roughness);
        float3 F = F_Schlick(F0, max(dot(H, V), 0.0));

        float3 kD = (1.0 - F) * (1.0 - metallic);
        float3 specular = D * G * F / (4.0 * NdotV * NdotL + 0.0001);
        float3 diffuse = kD * albedo / PI;

        combined_direct += (diffuse + specular) * NdotL
                          * lt.color.rgb * lt.color.w * atten;
    }

    // Ambient. The occlusion factor applies to the ambient (indirect) term at
    // the material's declared strength, never to direct light.
    float ao_eff = lerp(1.0, ao, mat.occlusion_strength);
    float3 ambient = albedo * ao_eff * sceneHeader[0].ambient_color.rgb * sceneHeader[0].ambient_color.a;

    // Emissive: the factor colors the emissive texture per the glTF
    // convention, and its .a carries the KHR_materials_emissive_strength
    // intensity. Without an emissive texture the factor alone glows.
    float2 uv_emissive = SlotUV(mat.uv_sets, 3u, uv0, uv1);
    float3 emissive_rgb = mat.emissive_factor.rgb;
    if (mat.emissive_texture != 0) {
        emissive_rgb *= allTextures[NonUniformResourceIndex(mat.emissive_texture)]
            .Sample(uv_emissive).rgb;
    }
    emissive_rgb *= mat.emissive_factor.a;

    float3 color = ambient + combined_direct + emissive_rgb;
    return float4(color, albedo_sample.a);
}

[shader("fragment")]
float4 main(PSInput input) : SV_TARGET {
    return Shade(input, input.texCoord, input.texCoord);
}

// UV1 variant: samples the per-slot selected set (UV0 or UV1) through the
// interpolated out-of-line UV1.
[shader("fragment")]
float4 main_uv1(PSInputUV1 input) : SV_TARGET {
    PSInput base;
    base.normal = input.normal;
    base.texCoord = input.texCoord;
    base.materialId = input.materialId;
    base.worldPos = input.worldPos;
    base.ormPacked = input.ormPacked;
    base.tangent = input.tangent;
    base.handedness = input.handedness;
    return Shade(base, input.texCoord, input.uv1);
}
