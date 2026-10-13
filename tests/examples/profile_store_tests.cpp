#include <gtest/gtest.h>

import std;
import std.compat;

import VulkanShared.Storage;
import VulkanShared.UserPaths;
import Examples.InfiniteRunner.Leaderboard.Account;
import Examples.InfiniteRunner.Account.ProfileStore;

namespace {

using Examples::InfiniteRunner::Account::ProfileStore;

std::filesystem::path ScratchBase() {
    static std::atomic<unsigned> counter{0};
    const auto nonce = static_cast<unsigned>(std::random_device{}());
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() /
        std::format("vkengine_profile_store_{}_{}", nonce, counter.fetch_add(1));
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base;
}

[[nodiscard]] std::string ReadTextFile(const std::filesystem::path& path) {
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

VulkanShared::Storage::Storage MakeStorage(const std::filesystem::path& base) {
    VulkanShared::UserPaths::Roots roots{};
    roots.persistent = base;
    roots.cache = base / "cache";
    roots.log = base / "logs";
    VulkanShared::Storage::Options options{};
    options.durable_writes = false;
    auto storage = VulkanShared::Storage::Storage::CreateWithRoots(std::move(roots), options);
    EXPECT_TRUE(storage.has_value());
    return std::move(*storage);
}

TEST(ProfileStoreTest, MultipleProfilesAreIsolated) {
    const std::filesystem::path base = ScratchBase();
    VulkanShared::Storage::Storage storage = MakeStorage(base);

    auto store = ProfileStore::Create(storage);
    ASSERT_TRUE(store.has_value());

    const auto& alice = store->AddProfile("alice", "Alice");
    const std::string alice_id = alice.id;
    ASSERT_FALSE(alice_id.empty());
    store->SetToken(alice_id, std::string(64, 'a'));
    store->RecordRun(0x1234, 1, 100, "Alice");
    EXPECT_EQ(store->LocalBest(0x1234), 100);
    ASSERT_EQ(store->LocalTop(0x1234, 5).size(), 1U);
    EXPECT_EQ(store->LocalTop(0x1234, 5).front().score, 100);

    const auto& bob = store->AddProfile("bob", "Bob");
    EXPECT_EQ(store->ActiveId(), bob.id);
    // Ids are the ImGui row identity, so identical display names must still get
    // distinct ids.
    EXPECT_NE(alice_id, bob.id);
    // Bob's board is separate from Alice's.
    EXPECT_EQ(store->LocalBest(0x1234), 0);

    ASSERT_TRUE(store->SetActive(alice_id));
    EXPECT_EQ(store->Active()->username, "alice");
    EXPECT_EQ(store->LocalBest(0x1234), 100);
    EXPECT_EQ(store->Active()->token, std::string(64, 'a'));

    ASSERT_TRUE(store->RemoveProfile(bob.id));
    EXPECT_EQ(store->Profiles().size(), 1U);

    std::error_code ec;
    std::filesystem::remove_all(base, ec);
}

TEST(ProfileStoreTest, PersistsAcrossReload) {
    const std::filesystem::path base = ScratchBase();
    VulkanShared::Storage::Storage storage = MakeStorage(base);

    std::string alice_id;
    {
        auto store = ProfileStore::Create(storage);
        ASSERT_TRUE(store.has_value());
        alice_id = store->AddProfile("alice", "Alice").id;
        store->SetToken(alice_id, std::string(64, 'b'));
        store->RecordRun(0x99, 7, 55, "Alice");
        ASSERT_TRUE(store->Save());
    }

    {
        auto store = ProfileStore::Create(storage);
        ASSERT_TRUE(store.has_value());
        ASSERT_EQ(store->Profiles().size(), 1U);
        EXPECT_EQ(store->ActiveId(), alice_id);
        ASSERT_NE(store->Active(), nullptr);
        EXPECT_EQ(store->Active()->display_name, "Alice");
        EXPECT_EQ(store->Active()->token, std::string(64, 'b'));
        EXPECT_EQ(store->LocalBest(0x99), 55);
    }

    std::error_code ec;
    std::filesystem::remove_all(base, ec);
}

TEST(ProfileStoreTest, IdsAreUniqueEvenWithIdenticalNames) {
    const std::filesystem::path base = ScratchBase();
    VulkanShared::Storage::Storage storage = MakeStorage(base);

    auto store = ProfileStore::Create(storage);
    ASSERT_TRUE(store.has_value());

    const std::string first = store->AddProfile("alice", "Alice").id;
    const std::string second = store->AddProfile("alice", "Alice").id;
    const std::string third = store->AddProfile("alice", "Alice").id;
    EXPECT_NE(first, second);
    EXPECT_NE(second, third);
    EXPECT_NE(first, third);

    std::error_code ec;
    std::filesystem::remove_all(base, ec);
}

TEST(ProfileStoreTest, DeleteRemovesProfileAndItsCredential) {
    const std::filesystem::path base = ScratchBase();
    VulkanShared::Storage::Storage storage = MakeStorage(base);

    std::string id;
    {
        auto store = ProfileStore::Create(storage);
        ASSERT_TRUE(store.has_value());
        id = store->AddProfile("alice", "Alice").id;
        store->SetToken(id, std::string(64, 'c'));
        store->RecordRun(0x1, 5, 10, "Alice");
        ASSERT_FALSE(id.empty());
        // The credential and the index must exist before we try to prove the
        // profile is gone from disk.
        ASSERT_TRUE(std::filesystem::exists(storage.SaveDir()));
        ASSERT_TRUE(std::filesystem::exists(base / "settings.json"));

        ASSERT_TRUE(store->RemoveProfile(id));
        EXPECT_TRUE(store->Profiles().empty());
        EXPECT_TRUE(store->ActiveId().empty());
    }
    {
        // The security property: no file left on disk still references the
        // deleted profile — neither its id (which the private credential slot
        // and the index both used to carry) nor its token. Enumerate the
        // store's own directories through the public SaveDir() accessor and the
        // harness persistent root instead of pinning the private slot naming.
        bool saw_any_file = false;
        std::error_code iter_ec;
        for (const auto& dir : {std::filesystem::path{base}, storage.SaveDir()}) {
            for (const std::filesystem::directory_entry& entry :
                 std::filesystem::directory_iterator(dir, iter_ec)) {
                if (!entry.is_regular_file(iter_ec)) {
                    continue;
                }
                saw_any_file = true;
                const std::string contents = ReadTextFile(entry.path());
                // The credential only ever lived in the private per-profile
                // slot, so it must be gone from every file, the recovery backup
                // included.
                EXPECT_EQ(contents.find(std::string(64, 'c')), std::string::npos)
                    << entry.path() << " still carries the deleted profile's token";
                // The id must be gone from the live files. settings.json.bak is
                // exempt: Storage keeps exactly the previous config generation
                // for hand-recovery, and the index carries ids, never secrets.
                if (entry.path() == storage.ConfigBackupPath()) {
                    continue;
                }
                EXPECT_EQ(contents.find(id), std::string::npos)
                    << entry.path() << " still references the deleted profile";
            }
        }
        ASSERT_TRUE(saw_any_file) << "the storage write nothing under " << base;

        auto store = ProfileStore::Create(storage);
        ASSERT_TRUE(store.has_value());
        EXPECT_TRUE(store->Profiles().empty());
    }

    std::error_code ec;
    std::filesystem::remove_all(base, ec);
}

TEST(ProfileStoreTest, DisplayNameAndLocalSettingsUpdate) {
    const std::filesystem::path base = ScratchBase();
    VulkanShared::Storage::Storage storage = MakeStorage(base);

    auto store = ProfileStore::Create(storage);
    ASSERT_TRUE(store.has_value());
    const std::string id = store->AddProfile("carol", "Carol").id;

    ASSERT_TRUE(store->UpdateDisplayName(id, "Carol Two"));
    Examples::InfiniteRunner::Leaderboard::LocalSettings local = store->Active()->local;
    // Flip the flags so the round trip is observable, whatever the defaults of
    // the board-owned settings struct are.
    local.leaderboard_best_per_account = false;
    local.leaderboard_only_mine = true;
    ASSERT_TRUE(store->UpdateLocal(id, local));

    ASSERT_NE(store->Active(), nullptr);
    EXPECT_EQ(store->Active()->display_name, "Carol Two");
    EXPECT_FALSE(store->Active()->local.leaderboard_best_per_account);
    EXPECT_TRUE(store->Active()->local.leaderboard_only_mine);

    std::error_code ec;
    std::filesystem::remove_all(base, ec);
}

} // namespace
