struct PSInput {
    [[vk::location(1)]] float2 texCoord : TEXCOORD0;
};

[shader("fragment")]
float4 main(PSInput input) : SV_TARGET {
    float check_size = 8.0;
    float2 uv_scaled = floor(input.texCoord * check_size);
    float checker = fmod(uv_scaled.x + uv_scaled.y, 2.0);
    return lerp(float4(0.2, 0.2, 0.2, 1.0), float4(0.8, 0.8, 0.8, 1.0), checker);
}
