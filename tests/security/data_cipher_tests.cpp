#include <gtest/gtest.h>

#include "engine/security/cipher_format.hpp"

import std;
import std.compat;

import VulkanEngine.DataCipher;

namespace {

using VulkanEngine::Security::CipherVariant;
using VulkanEngine::Security::KeyRing;
using VulkanEngine::Security::kKeyBytes;

namespace fmt = vkengine::security::format;

constexpr std::uint8_t kKeyId = 1;

KeyRing MakeRing() {
    KeyRing ring;
    std::array<std::byte, kKeyBytes> key{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<std::byte>(i + 1);
    }
    ring.Add(kKeyId, key);
    return ring;
}

std::span<const std::byte> AsBytes(const std::string& text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

std::string AsString(std::span<const std::byte> bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

constexpr std::array kVariants = {
    CipherVariant::XChaCha20Poly1305,
    CipherVariant::ChaCha20Poly1305Ietf,
};

TEST(DataCipherTest, RoundTripsEveryVariant) {
    const KeyRing ring = MakeRing();
    const std::string secret = "127.0.0.1:7777";

    for (const CipherVariant variant : kVariants) {
        const std::vector<std::byte> sealed = SealRandom(ring, variant, kKeyId, AsBytes(secret));
        ASSERT_FALSE(sealed.empty()) << "variant " << static_cast<int>(variant);

        const std::optional<std::vector<std::byte>> opened = Open(ring, sealed);
        ASSERT_TRUE(opened.has_value()) << "variant " << static_cast<int>(variant);
        EXPECT_EQ(AsString(*opened), secret);
    }
}

TEST(DataCipherTest, ExplicitNonceIsReproducible) {
    const KeyRing ring = MakeRing();
    const std::string secret = "leaderboard.example:9000";

    for (const CipherVariant variant : kVariants) {
        const std::vector<std::byte> nonce = RandomNonce(variant);
        const std::vector<std::byte> a = Seal(ring, variant, kKeyId, nonce, AsBytes(secret));
        const std::vector<std::byte> b = Seal(ring, variant, kKeyId, nonce, AsBytes(secret));
        EXPECT_EQ(a, b) << "variant " << static_cast<int>(variant);
    }
}

TEST(DataCipherTest, TamperedCiphertextIsRejected) {
    const KeyRing ring = MakeRing();
    std::vector<std::byte> sealed = SealRandom(ring, CipherVariant::XChaCha20Poly1305, kKeyId, AsBytes("secret"));

    const std::size_t last = sealed.size() - 1;
    sealed[last] = static_cast<std::byte>(static_cast<unsigned>(sealed[last]) ^ 0x01U);

    EXPECT_FALSE(Open(ring, sealed).has_value());
}

TEST(DataCipherTest, TamperedHeaderVariantIsRejected) {
    const KeyRing ring = MakeRing();
    std::vector<std::byte> sealed = SealRandom(ring, CipherVariant::XChaCha20Poly1305, kKeyId, AsBytes("secret"));
    ASSERT_GE(sealed.size(), fmt::kHeaderFixedBytes);

    // Envelope layout (cipher_format.hpp): the fixed header ends with the
    // key_id byte and a reserved byte; the cipher-variant word is the u16
    // directly before them — i.e. the two bytes of the fixed header that
    // precede the key_id. Derive the poke positions from the layout constant.
    const std::size_t kKeyIdPos = fmt::kHeaderFixedBytes - 2;
    const std::size_t kVariantU16Pos = kKeyIdPos - 2;

    // Claim the IETF variant via the format's own serializer while the payload
    // is XChaCha: Open() must then select the wrong AEAD and fail on the tag.
    fmt::WriteU16(
        reinterpret_cast<std::uint8_t*>(sealed.data()) + kVariantU16Pos,
        static_cast<std::uint16_t>(fmt::CipherVariant::ChaCha20Poly1305Ietf));

    // The poke actually rewrote the variant word the header API reads.
    ASSERT_TRUE(VulkanEngine::Security::PeekVariant(sealed).has_value());
    EXPECT_EQ(*VulkanEngine::Security::PeekVariant(sealed), CipherVariant::ChaCha20Poly1305Ietf);

    EXPECT_FALSE(Open(ring, sealed).has_value());
}

TEST(DataCipherTest, TruncatedEnvelopeIsRejected) {
    const KeyRing ring = MakeRing();
    std::vector<std::byte> sealed = SealRandom(ring, CipherVariant::XChaCha20Poly1305, kKeyId, AsBytes("secret"));
    sealed.resize(sealed.size() - 1);

    EXPECT_FALSE(Open(ring, sealed).has_value());
}

TEST(DataCipherTest, MissingKeyIsRejected) {
    const KeyRing ring = MakeRing();
    const std::vector<std::byte> sealed = SealRandom(ring, CipherVariant::XChaCha20Poly1305, kKeyId, AsBytes("secret"));

    KeyRing other;
    std::array<std::byte, kKeyBytes> key{};
    other.Add(7, key);
    EXPECT_FALSE(Open(other, sealed).has_value());
}

TEST(DataCipherTest, AadMismatchIsRejected) {
    const KeyRing ring = MakeRing();
    const std::vector<std::byte> sealed = SealRandom(
        ring, CipherVariant::XChaCha20Poly1305, kKeyId, AsBytes("secret"), AsBytes("endpoint"));

    EXPECT_FALSE(Open(ring, sealed, AsBytes("other")).has_value());
    EXPECT_TRUE(Open(ring, sealed, AsBytes("endpoint")).has_value());
}

TEST(DataCipherTest, PeekReportsEnvelopeMetadata) {
    const KeyRing ring = MakeRing();
    const std::vector<std::byte> sealed = SealRandom(
        ring, CipherVariant::ChaCha20Poly1305Ietf, kKeyId, AsBytes("secret"));

    ASSERT_TRUE(VulkanEngine::Security::PeekKeyId(sealed).has_value());
    EXPECT_EQ(*VulkanEngine::Security::PeekKeyId(sealed), kKeyId);
    ASSERT_TRUE(VulkanEngine::Security::PeekVariant(sealed).has_value());
    EXPECT_EQ(*VulkanEngine::Security::PeekVariant(sealed), CipherVariant::ChaCha20Poly1305Ietf);
}

} // namespace
