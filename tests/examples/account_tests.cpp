#include <gtest/gtest.h>

import std;
import std.compat;

import VulkanEngine.PasswordHash;
import Examples.InfiniteRunner.Leaderboard.Account;
import Examples.InfiniteRunner.Leaderboard.AccountStore;
import Examples.InfiniteRunner.Leaderboard.Config;

namespace {

using namespace Examples::InfiniteRunner::Leaderboard;

VulkanEngine::Security::Argon2Params TestArgon2() {
    VulkanEngine::Security::Argon2Params params{};
    params.blocks = 64;
    params.passes = 2;
    params.lanes = 1;
    params.hash_bytes = 32;
    params.salt_bytes = 16;
    return params;
}

AccountStoreOptions MemoryStore() {
    AccountStoreOptions options{};
    options.argon2 = TestArgon2();
    return options;
}

TEST(UsernameTest, CanonicalizesAndValidates) {
    EXPECT_EQ(CanonicalizeUsername("  Alice  "), std::optional<std::string>{"alice"});
    // The length boundaries come from the named constants: one below the
    // minimum is rejected, the minimum is accepted, the maximum is accepted, and
    // one above the maximum is rejected.
    EXPECT_FALSE(CanonicalizeUsername(std::string(kUsernameMinLength - 1, 'a')).has_value());
    EXPECT_TRUE(CanonicalizeUsername(std::string(kUsernameMinLength, 'a')).has_value());
    EXPECT_TRUE(CanonicalizeUsername(std::string(kUsernameMaxLength, 'a')).has_value());
    EXPECT_FALSE(CanonicalizeUsername(std::string(kUsernameMaxLength + 1, 'a')).has_value());
    EXPECT_FALSE(CanonicalizeUsername("1alice").has_value());      // must start with a letter
    EXPECT_FALSE(CanonicalizeUsername("al ice").has_value());      // space
    EXPECT_FALSE(CanonicalizeUsername("alice!").has_value());      // punctuation
}

TEST(DisplayNameTest, SanitizesDisplayNames) {
    EXPECT_EQ(SanitizeDisplayName("  Alice  "), std::optional<std::string>{"Alice"});
    EXPECT_FALSE(SanitizeDisplayName("").has_value());
    EXPECT_FALSE(SanitizeDisplayName("bad\nname").has_value());
    // The display-name length limit is the named constant: exactly at it is
    // accepted, one past it is rejected.
    EXPECT_TRUE(SanitizeDisplayName(std::string(kDisplayNameMaxLength, 'A')).has_value());
    EXPECT_FALSE(SanitizeDisplayName(std::string(kDisplayNameMaxLength + 1, 'A')).has_value());
}

TEST(AccountStoreTest, RegisterLoginRoundTrip) {
    AccountStore store(MemoryStore());
    const auto registered = store.Register("Alice", "Alice");
    ASSERT_EQ(registered.status, SyncStatus::Ok);
    EXPECT_EQ(registered.display_name, "Alice");
    EXPECT_EQ(registered.token.size(), kTokenHexLength);

    const auto login = store.Login("alice", registered.token);
    EXPECT_EQ(login.status, SyncStatus::Ok);
    EXPECT_EQ(login.account.id, registered.user_id);
    EXPECT_EQ(login.account.username, "alice");
}

TEST(AccountStoreTest, RejectsDuplicateUsernameIgnoringCase) {
    AccountStore store(MemoryStore());
    ASSERT_EQ(store.Register("alice", "Alice").status, SyncStatus::Ok);
    EXPECT_EQ(store.Register("ALICE", "Other").status, SyncStatus::UsernameTaken);
}

TEST(AccountStoreTest, RejectsWrongToken) {
    AccountStore store(MemoryStore());
    const auto registered = store.Register("alice", "Alice");
    ASSERT_EQ(registered.status, SyncStatus::Ok);

    std::string wrong = registered.token;
    wrong[0] = wrong[0] == '0' ? '1' : '0';
    EXPECT_EQ(store.Login("alice", wrong).status, SyncStatus::InvalidToken);
    EXPECT_EQ(store.Login("alice", "not-hex").status, SyncStatus::InvalidToken);
    EXPECT_EQ(store.Login("nobody", registered.token).status, SyncStatus::UnknownAccount);
}

TEST(AccountStoreTest, RunIdsAreIdempotent) {
    AccountStore store(MemoryStore());
    const auto registered = store.Register("alice", "Alice");
    ASSERT_EQ(registered.status, SyncStatus::Ok);

    EXPECT_EQ(store.RecordRun(registered.user_id, 1234), AccountStore::ScoreResult::Accepted);
    EXPECT_EQ(store.RecordRun(registered.user_id, 1234), AccountStore::ScoreResult::Duplicate);
    EXPECT_EQ(store.RecordRun(registered.user_id, 5678), AccountStore::ScoreResult::Accepted);
    EXPECT_EQ(store.RecordRun(999999, 1), AccountStore::ScoreResult::UnknownAccount);
}

TEST(AccountStoreTest, RenameCarriesAReason) {
    AccountStore store(MemoryStore());
    const auto registered = store.Register("alice", "Alice");
    ASSERT_EQ(registered.status, SyncStatus::Ok);

    const auto updated = store.Rename(registered.user_id, "Alice Cooper");
    EXPECT_EQ(updated.status, SyncStatus::Ok);
    EXPECT_EQ(updated.account.display_name, "Alice Cooper");

    const auto rejected = store.Rename(registered.user_id, "bad\nname");
    EXPECT_EQ(rejected.status, SyncStatus::Rejected);
    EXPECT_FALSE(rejected.reason.empty());
}

// A hidden legacy (v2) account stays anonymous on the board, and renaming it
// must not reveal it.
TEST(AccountStoreTest, LegacyVisibilityAndRenameDoNotReveal) {
    AccountStore store(MemoryStore());
    const auto registered = store.Register("alice", "Alice");
    ASSERT_EQ(registered.status, SyncStatus::Ok);

    EXPECT_FALSE(store.IsHidden(registered.user_id));
    ASSERT_TRUE(store.ShownName(registered.user_id).has_value());
    EXPECT_EQ(*store.ShownName(registered.user_id), "Alice");
    EXPECT_FALSE(store.ShownName(999).has_value());

    ASSERT_EQ(store.UpdateLegacy(registered.user_id, "Alice", /*show_on_leaderboard=*/false).status,
              SyncStatus::Ok);
    EXPECT_TRUE(store.IsHidden(registered.user_id));
    EXPECT_EQ(*store.ShownName(registered.user_id), std::string{});

    ASSERT_EQ(store.Rename(registered.user_id, "Alice Cooper").status, SyncStatus::Ok);
    EXPECT_TRUE(store.IsHidden(registered.user_id));
    EXPECT_EQ(*store.ShownName(registered.user_id), std::string{});
}

TEST(AccountStoreTest, RejectsBlockedDisplayName) {
    AccountStoreOptions options = MemoryStore();
    options.name_policy.blocked = {"admin"};
    AccountStore store(options);

    const auto result = store.Register("alice", "The Admin");
    EXPECT_EQ(result.status, SyncStatus::Rejected);
    EXPECT_FALSE(result.reason.empty());
}

// The same content policy applies to the permanent username.
TEST(AccountStoreTest, RejectsBlockedUsername) {
    AccountStoreOptions options = MemoryStore();
    options.name_policy.blocked = {"dick"};
    AccountStore store(options);

    const auto blocked = store.Register("dick", "Fine Name");
    EXPECT_EQ(blocked.status, SyncStatus::Rejected);
    EXPECT_FALSE(blocked.reason.empty());
    // A structurally invalid username is still reported as invalid rather than
    // as a policy refusal.
    EXPECT_EQ(store.Register("1bad", "Fine Name").status, SyncStatus::InvalidUsername);
}

TEST(NamePolicyTest, NormalizationFoldsCaseInvisibleAndWhitespace) {
    EXPECT_EQ(NormalizeNameForPolicy("  Alice  "), "alice");
    EXPECT_EQ(NormalizeNameForPolicy("A\t B"), "a b");
    // Zero-width space (U+200B) and soft hyphen (U+00AD) are removed.
    EXPECT_EQ(NormalizeNameForPolicy(std::string{"b"} + "\xE2\x80\x8B" + "ad"), "bad");
    EXPECT_EQ(NormalizeNameForPolicy("bad\xC2\xADword"), "badword");
}

// Separators inside a token are erased before matching, so obfuscation does not
// hide a blocked word.
TEST(NamePolicyTest, SeparatorsDoNotHideABlockedWord) {
    NamePolicy policy;
    policy.blocked = {"bad"};
    EXPECT_TRUE(policy.Check("b.a.d").has_value());
    EXPECT_TRUE(policy.Check("b_a_d").has_value());
    EXPECT_TRUE(policy.Check("xXb.a.dXx").has_value());
    // Whitespace is a token boundary, so this is deliberately not matched.
    EXPECT_FALSE(policy.Check("d i c k").has_value());
    EXPECT_FALSE(policy.Check("clean name").has_value());
}

// One allow prefix covers every variant, and the exemption applies only to the
// span it covers, so padding cannot smuggle a blocked word past it.
TEST(NamePolicyTest, AllowPrefixCoversVariantsWithoutOpeningAHole) {
    NamePolicy policy;
    policy.blocked = {"dick"};
    policy.allowed = {"dickin"};
    EXPECT_FALSE(policy.Check("dickinson").has_value());
    EXPECT_FALSE(policy.Check("Dickinger").has_value());
    EXPECT_TRUE(policy.Check("dick").has_value());
    EXPECT_TRUE(policy.Check("xxdickxx").has_value());
    // The second "dick" is not covered by the allowed span.
    EXPECT_TRUE(policy.Check("dickindick").has_value());
    EXPECT_TRUE(policy.Check("dick dickin").has_value());
}

// Common names that contain a blocked substring, with the allow roots a real
// policy file would carry.
TEST(NamePolicyTest, CommonNamesContainingBlockedSubstrings) {
    NamePolicy policy;
    policy.blocked = {"cock", "rape", "spic", "anus", "bugger"};
    // An allow entry only helps if it is actually a substring of the name, so
    // "spicy" needs its own root rather than being covered by "spice".
    policy.allowed = {"cocktail", "peacock", "grape", "spice", "spicy", "janus", "debugger"};
    EXPECT_FALSE(policy.Check("Peacock").has_value());
    EXPECT_FALSE(policy.Check("cocktail").has_value());
    EXPECT_FALSE(policy.Check("Grapefruit").has_value());
    EXPECT_FALSE(policy.Check("Spicy").has_value());
    EXPECT_FALSE(policy.Check("Janus").has_value());
    EXPECT_FALSE(policy.Check("debugger").has_value());
    EXPECT_TRUE(policy.Check("cock").has_value());
    EXPECT_TRUE(policy.Check("rape").has_value());
    EXPECT_TRUE(policy.Check("spic").has_value());
}

TEST(NamePolicyTest, LoadsFromFile) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vkengine_names_policy_test.txt";
    {
        std::ofstream out(path);
        out << "# a comment\n";
        out << "block BadWord\n";
        out << "block badword\n";  // duplicate after normalization
        out << "block f.u.c.k\n";  // separators are compacted away
        out << "allow Badworder\n";
        out << "bogus directive\n";
        out << "block two words\n";  // multi-word: cannot match, skipped
    }

    const NamePolicy policy = LoadNamePolicy(path);
    // Per-entry membership plus behavior instead of pinning the exact blocked
    // vector. The count still covers the semantics of the input: two of the
    // three block directives survive (case-dedupe drops one) and the separator
    // compaction turns "f.u.c.k" into a single entry.
    EXPECT_EQ(policy.blocked.size(), 2U);
    const auto contains = [&policy](std::string_view entry) {
        return std::find(policy.blocked.begin(), policy.blocked.end(), entry) !=
               policy.blocked.end();
    };
    EXPECT_TRUE(contains("badword"));
    EXPECT_TRUE(contains("fuck"));
    // Every loaded block entry must actually fire as written.
    for (const std::string& entry : policy.blocked) {
        EXPECT_TRUE(policy.Check(entry).has_value()) << entry;
    }
    EXPECT_TRUE(policy.allowed.contains("badworder"));
    EXPECT_FALSE(policy.Check("badworder").has_value()); // allowed root covers the block
    EXPECT_TRUE(policy.Check("badword").has_value());

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(AccountStoreTest, PersistsAcrossReload) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vkengine_account_store_test.json";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    std::string token;
    UserId user_id = 0;
    {
        AccountStoreOptions options = MemoryStore();
        options.path = path;
        AccountStore store(options);
        const auto registered = store.Register("alice", "Alice");
        ASSERT_EQ(registered.status, SyncStatus::Ok);
        token = registered.token;
        user_id = registered.user_id;
    }
    {
        AccountStoreOptions options = MemoryStore();
        options.path = path;
        AccountStore store(options);
        EXPECT_EQ(store.AccountCount(), 1U);
        const auto login = store.Login("alice", token);
        EXPECT_EQ(login.status, SyncStatus::Ok);
        EXPECT_EQ(login.account.id, user_id);
    }
    std::filesystem::remove(path, ec);
}

TEST(SessionStoreTest, IssuesResolvesAndRevokes) {
    SessionStoreOptions options{};
    options.ttl = std::chrono::hours(1);
    SessionStore sessions(options);

    const auto issued = sessions.Issue(42);
    ASSERT_FALSE(issued.token.empty());
    EXPECT_EQ(sessions.Resolve(issued.token), std::optional<UserId>{42});
    EXPECT_FALSE(sessions.Resolve("deadbeef").has_value());

    sessions.Revoke(issued.token);
    EXPECT_FALSE(sessions.Resolve(issued.token).has_value());
}

} // namespace
