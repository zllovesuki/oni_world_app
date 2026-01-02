#ifndef EMSCRIPTEN

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <signal.h>

#include "WorldGen.hpp"
#include "config.h"

namespace {

constexpr int32_t kSeedMin = 1;
constexpr int32_t kSeedMax = 2147483647;

struct SeedSearchConfig {
    std::string clusterPrefix = "CER-A";
    // Match the repo UI default (`src/jsUtils/toolbar.tsx` uses `mixings=9769375`).
    std::string mixingCode = SettingsCache::BinaryToBase36(9769375);

    // Trait keys are SettingsCache keys (e.g. "traits/GeoActive").
    std::vector<std::string> requireTraits = {"traits/MetalRich", "traits/Geodes",
                                              "traits/FrozenCore"};
    // User requirement: exclude Geodormant and Volcanic Activity (explicitly
    // configured here as "traits/Volcanoes"). Override via CLI if desired.
    std::vector<std::string> forbidTraits = {"traits/Geodormant", "traits/Volcanoes"};
    bool exactTraits = false;

    int32_t seedStart = kSeedMin;
    int32_t seedEnd = kSeedMax;

    int threads = 0;
    int32_t chunkSize = 100000;
    int progressEverySeconds = 1;

    double verticalWeight = 2.0;
    int32_t waterMaxDist = 80;
    int32_t metalMaxDist = 120;
    int32_t oilMaxPairDist = 100;

    double wWater = 1000.0;
    double wMetal = 200.0;
    double wOil = 100.0;
    double wCenter = 1.0;

    std::string checkpointPath = "seed_search_checkpoint.json";
    std::string outputPath = "seed_search_results.jsonl";
    bool resume = true;
};

struct SeedEval {
    int32_t seed = 0;
    std::string code;

    Vector2i worldSize{};
    Vector2i start{};

    std::vector<std::string> traits;

    // Key geyser positions (all y-flipped to match UI coordinate space).
    std::vector<Vector2i> water;
    std::vector<Vector2i> copper;
    std::vector<Vector2i> iron;
    std::vector<Vector2i> gold;
    std::vector<Vector2i> oilReservoirs;

    int32_t distWater = INT32_MAX;
    int32_t distMetalAny = INT32_MAX;
    int32_t oilMaxPairDist = INT32_MAX;
    int32_t centerOffsetX = INT32_MAX;

    double score = 0.0;
};

struct WorkItem {
    uint64_t chunkId = 0;
    int32_t seedBegin = 0;      // inclusive
    int32_t seedEndExclusive = 0; // exclusive
};

struct CompletedChunk {
    uint64_t chunkId = 0;
};

class WorkQueue
{
public:
    void Push(WorkItem item)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_closed) {
                return;
            }
            m_queue.push(std::move(item));
        }
        m_cv.notify_one();
    }

    // Returns nullopt if closed and empty.
    std::optional<WorkItem> Pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&] { return m_closed || !m_queue.empty(); });
        if (m_queue.empty()) {
            return std::nullopt;
        }
        WorkItem item = std::move(m_queue.front());
        m_queue.pop();
        return item;
    }

    void Close()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closed = true;
        }
        m_cv.notify_all();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<WorkItem> m_queue;
    bool m_closed = false;
};

class CompletionQueue
{
public:
    void Push(CompletedChunk item)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_closed) {
                return;
            }
            m_queue.push(std::move(item));
        }
        m_cv.notify_one();
    }

    // Returns nullopt if closed and empty.
    std::optional<CompletedChunk> Pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&] { return m_closed || !m_queue.empty(); });
        if (m_queue.empty()) {
            return std::nullopt;
        }
        CompletedChunk item = std::move(m_queue.front());
        m_queue.pop();
        return item;
    }

    void Close()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closed = true;
        }
        m_cv.notify_all();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<CompletedChunk> m_queue;
    bool m_closed = false;
};

static std::atomic<bool> g_stop{false};

static void OnSignal(int)
{
    g_stop.store(true, std::memory_order_relaxed);
}

static std::string JsonEscape(std::string_view s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if ((unsigned char)c < 0x20) {
                char buf[7];
                snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                out += buf;
            } else {
                out += c;
            }
            break;
        }
    }
    return out;
}

static bool WriteFileAtomic(const std::filesystem::path &path,
                            const std::string &content)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }
        out.write(content.data(), (std::streamsize)content.size());
        out.flush();
        if (!out.good()) {
            return false;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        // Fallback: try to replace existing.
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
    return !ec;
}

static std::optional<std::string> ReadFileToString(const std::filesystem::path &p)
{
    std::ifstream in(p, std::ios::binary);
    if (!in.is_open()) {
        return std::nullopt;
    }
    std::string s;
    in.seekg(0, std::ios::end);
    s.resize((size_t)in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(s.data(), (std::streamsize)s.size());
    if (!in.good()) {
        return std::nullopt;
    }
    return s;
}

static int32_t WeightedTileDist(Vector2i a, Vector2i b, double verticalWeight)
{
    auto dx = std::abs(a.x - b.x);
    auto dy = std::abs(a.y - b.y);
    double d = (double)dx + verticalWeight * (double)dy;
    if (d > (double)INT32_MAX) {
        return INT32_MAX;
    }
    return (int32_t)std::llround(d);
}

static int32_t MinDist(const Vector2i &from, const std::vector<Vector2i> &to,
                       double verticalWeight)
{
    int32_t best = INT32_MAX;
    for (auto &p : to) {
        best = std::min(best, WeightedTileDist(from, p, verticalWeight));
    }
    return best;
}

static int32_t MaxPairwiseDist(const std::vector<Vector2i> &pts,
                               double verticalWeight)
{
    if (pts.size() < 2) {
        return 0;
    }
    int32_t best = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = i + 1; j < pts.size(); ++j) {
            best = std::max(best, WeightedTileDist(pts[i], pts[j], verticalWeight));
        }
    }
    return best;
}

static std::string BuildCode(std::string_view clusterPrefix, int32_t seed,
                             std::string_view mixingCode)
{
    std::string code;
    code.reserve(clusterPrefix.size() + 64);
    code.append(clusterPrefix);
    code.push_back('-');
    code.append(std::to_string(seed));
    code.append("-0-D3-");
    code.append(mixingCode);
    return code;
}

static bool HasTrait(const std::vector<const WorldTrait *> &traits,
                     std::string_view key)
{
    for (auto *t : traits) {
        if (t && t->filePath == key) {
            return true;
        }
    }
    return false;
}

static void ExtractTraits(const std::vector<const WorldTrait *> &traits,
                          std::vector<std::string> &out)
{
    out.clear();
    out.reserve(traits.size());
    for (auto *t : traits) {
        if (!t) {
            continue;
        }
        out.push_back(t->filePath);
    }
}

struct ThreadState {
    SettingsCache settings;
    std::vector<MixingConfig> baseMixConfigs;
    std::vector<World *> worlds;
    World *startWorld = nullptr;
    int startWorldIndex = 0;
};

static bool InitThreadState(ThreadState &st, std::string_view zipBytes,
                            const SeedSearchConfig &cfg,
                            std::string &error)
{
    if (!st.settings.LoadSettingsCache(zipBytes)) {
        error = "LoadSettingsCache failed (data.zip unreadable or invalid).";
        return false;
    }

    // Use CoordinateChanged once to:
    // - select the cluster by coordinatePrefix
    // - apply the mixing code (resets mixConfigs min/max/level)
    std::string probe = BuildCode(cfg.clusterPrefix, 1, cfg.mixingCode);
    if (!st.settings.CoordinateChanged(probe, st.settings) ||
        st.settings.cluster == nullptr) {
        error = "CoordinateChanged failed for probe code: " + probe;
        return false;
    }

    st.baseMixConfigs = st.settings.mixConfigs;

    // Build worlds list for this cluster (Ceres base game: only 1 world).
    st.worlds.clear();
    auto &cluster = *st.settings.cluster;
    for (auto &wp : cluster.worldPlacements) {
        auto itr = st.settings.worlds.find(wp.world);
        if (itr == st.settings.worlds.end()) {
            error = "World not found: " + wp.world;
            return false;
        }
        itr->second.locationType = wp.locationType;
        st.worlds.push_back(&itr->second);
    }
    if (st.worlds.empty()) {
        error = "Cluster has no worldPlacements.";
        return false;
    }
    if (st.worlds.size() == 1) {
        st.worlds[0]->locationType = LocationType::StartWorld;
        st.startWorldIndex = 0;
        st.startWorld = st.worlds[0];
    } else {
        st.startWorldIndex = cluster.startWorldIndex;
        st.startWorld = st.worlds[st.startWorldIndex];
        if (st.startWorld == nullptr) {
            error = "Start world pointer is null.";
            return false;
        }
    }
    return true;
}

static std::optional<SeedEval> EvaluateSeed(ThreadState &st, int32_t baseSeed,
                                            const SeedSearchConfig &cfg)
{
    // Required geyser type indices (WorldGen::GetGeysers).
    constexpr int kGeyserCleanWater = 2; // "hot_water"
    constexpr int kGeyserCopperVolcano = 16;
    constexpr int kGeyserIronVolcano = 17;
    constexpr int kGeyserGoldVolcano = 18;
    constexpr int kGeyserOilReservoir = 27; // poi/oil/*

    // Early trait rejection (no world generation).
    int32_t traitSeed = baseSeed + st.startWorldIndex;
    st.settings.seed = traitSeed;
    auto pickedTraits = st.settings.GetRandomTraits(*st.startWorld);

    std::set<std::string_view> traitSet;
    for (auto *t : pickedTraits) {
        if (t) {
            traitSet.emplace(t->filePath);
        }
    }
    for (auto &req : cfg.requireTraits) {
        if (!traitSet.contains(req)) {
            return std::nullopt;
        }
    }
    for (auto &forbid : cfg.forbidTraits) {
        if (traitSet.contains(forbid)) {
            return std::nullopt;
        }
    }
    if (cfg.exactTraits && traitSet.size() != cfg.requireTraits.size()) {
        return std::nullopt;
    }

    // Reset mixing configs (DoSubworldMixing mutates mixConfigs' min/max).
    st.settings.mixConfigs = st.baseMixConfigs;

    // Apply mixing (resets per-world cached state).
    st.settings.seed = baseSeed;
    st.settings.DoSubworldMixing(st.worlds);

    // Apply traits to start world and generate.
    st.settings.seed = traitSeed;
    for (auto *t : pickedTraits) {
        if (t) {
            st.startWorld->ApplayTraits(*t, st.settings);
        }
    }
    WorldGen worldGen(*st.startWorld, st.settings);
    if (!worldGen.GenerateOverworld()) {
        return std::nullopt;
    }

    SeedEval out;
    out.seed = baseSeed;
    out.code = BuildCode(cfg.clusterPrefix, baseSeed, cfg.mixingCode);
    out.worldSize = worldGen.GetWorldSize();
    {
        // Prefer the actual "StartLocation" leaf site (used by TemplateSpawning),
        // not the root site list.
        std::optional<Vector2i> startLeaf;
        for (auto &root : worldGen.GetSites()) {
            if (!root.children) {
                continue;
            }
            for (auto &child : *root.children) {
                if (child.tags.contains("StartLocation")) {
                    startLeaf = Vector2i{child.x, child.y};
                    break;
                }
            }
            if (startLeaf) {
                break;
            }
        }
        Vector2i s = startLeaf.value_or(worldGen.GetStarting());
        out.start = {s.x, out.worldSize.y - s.y};
    }
    ExtractTraits(pickedTraits, out.traits);

    // Geysers and POIs
    int32_t globalWorldSeed =
        baseSeed + (int32_t)st.settings.cluster->worldPlacements.size() - 1;
    auto geysers = worldGen.GetGeysers(globalWorldSeed);
    out.oilReservoirs.clear();
    for (auto &g : geysers) {
        Vector2i pos{g.x, out.worldSize.y - g.y};
        switch (g.z) {
        case kGeyserCleanWater:
            out.water.push_back(pos);
            break;
        case kGeyserCopperVolcano:
            out.copper.push_back(pos);
            break;
        case kGeyserIronVolcano:
            out.iron.push_back(pos);
            break;
        case kGeyserGoldVolcano:
            out.gold.push_back(pos);
            break;
        case kGeyserOilReservoir:
            out.oilReservoirs.push_back(pos);
            break;
        default:
            break;
        }
    }

    // Hard geyser existence requirements.
    if (out.water.empty() || out.copper.empty() || out.iron.empty() || out.gold.empty()) {
        return std::nullopt;
    }

    // Distances / clustering.
    out.distWater = MinDist(out.start, out.water, cfg.verticalWeight);
    std::vector<Vector2i> anyMetal;
    anyMetal.reserve(out.copper.size() + out.iron.size() + out.gold.size());
    anyMetal.insert(anyMetal.end(), out.copper.begin(), out.copper.end());
    anyMetal.insert(anyMetal.end(), out.iron.begin(), out.iron.end());
    anyMetal.insert(anyMetal.end(), out.gold.begin(), out.gold.end());
    out.distMetalAny = MinDist(out.start, anyMetal, cfg.verticalWeight);
    out.oilMaxPairDist = MaxPairwiseDist(out.oilReservoirs, cfg.verticalWeight);

    int32_t midX = out.worldSize.x / 2;
    out.centerOffsetX = std::abs(out.start.x - midX);

    // Hard location thresholds (configurable).
    if (out.distWater > cfg.waterMaxDist) {
        return std::nullopt;
    }
    if (out.distMetalAny > cfg.metalMaxDist) {
        return std::nullopt;
    }
    if (out.oilReservoirs.size() < 2) {
        return std::nullopt;
    }
    if (out.oilMaxPairDist > cfg.oilMaxPairDist) {
        return std::nullopt;
    }

    // Score (higher is better).
    double score = 0.0;
    score += cfg.wWater * (double)(cfg.waterMaxDist - out.distWater);
    score += cfg.wMetal * (double)(cfg.metalMaxDist - out.distMetalAny);
    score += cfg.wOil * (double)(cfg.oilMaxPairDist - out.oilMaxPairDist);
    score += cfg.wCenter * (double)(-out.centerOffsetX);
    out.score = score;

    return out;
}

static std::string ToCheckpointJson(const SeedSearchConfig &cfg,
                                    int32_t nextSeed,
                                    uint64_t seedsTried,
                                    uint64_t seedsMatched,
                                    double seedsPerSec,
                                    uint64_t nextChunkId)
{
    auto arrStr = [](const std::vector<std::string> &v) {
        std::ostringstream s;
        s << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i)
                s << ",";
            s << "\"" << JsonEscape(v[i]) << "\"";
        }
        s << "]";
        return s.str();
    };

    std::ostringstream s;
    s << "{";
    s << "\"version\":1,";
    s << "\"cluster\":\"" << JsonEscape(cfg.clusterPrefix) << "\",";
    s << "\"mix\":\"" << JsonEscape(cfg.mixingCode) << "\",";
    s << "\"require_traits\":" << arrStr(cfg.requireTraits) << ",";
    s << "\"forbid_traits\":" << arrStr(cfg.forbidTraits) << ",";
    s << "\"exact_traits\":" << (cfg.exactTraits ? "true" : "false") << ",";
    s << "\"seed_start\":" << cfg.seedStart << ",";
    s << "\"seed_end\":" << cfg.seedEnd << ",";
    s << "\"chunk_size\":" << cfg.chunkSize << ",";
    s << "\"threads\":" << cfg.threads << ",";
    s << "\"vertical_weight\":" << cfg.verticalWeight << ",";
    s << "\"water_max\":" << cfg.waterMaxDist << ",";
    s << "\"metal_max\":" << cfg.metalMaxDist << ",";
    s << "\"oil_max_pair\":" << cfg.oilMaxPairDist << ",";
    s << "\"next_seed\":" << nextSeed << ",";
    s << "\"next_chunk_id\":" << nextChunkId << ",";
    s << "\"seeds_tried\":" << seedsTried << ",";
    s << "\"seeds_matched\":" << seedsMatched << ",";
    s << "\"seeds_per_sec\":" << std::fixed << std::setprecision(2) << seedsPerSec;
    s << "}";
    return s.str();
}

static std::optional<int32_t> ParseInt32(std::string_view s)
{
    try {
        long long v = std::stoll(std::string(s));
        if (v < INT32_MIN || v > INT32_MAX) {
            return std::nullopt;
        }
        return (int32_t)v;
    } catch (...) {
        return std::nullopt;
    }
}

static std::optional<double> ParseDouble(std::string_view s)
{
    try {
        return std::stod(std::string(s));
    } catch (...) {
        return std::nullopt;
    }
}

static void PrintSearchUsage()
{
    SeedSearchConfig d;
    d.threads = (int)std::max(1u, std::thread::hardware_concurrency());
    std::cerr
        << "Usage:\n"
        << "  oniWorldApp search [options]\n\n"
        << "Options (defaults in parentheses):\n"
        << "  --start N              start seed (" << kSeedMin << ")\n"
        << "  --end N                end seed (" << kSeedMax << ")\n"
        << "  --mix CODE             mixing code (" << d.mixingCode << ")\n"
        << "  --require-trait KEY    require world trait (repeatable; defaults: MetalRich, Geodes, FrozenCore)\n"
        << "  --forbid-trait KEY     forbid world trait (repeatable; defaults: Volcanoes, Geodormant)\n"
        << "  --exact-traits         require ONLY the required traits (default: off)\n"
        << "  --threads N            worker threads (" << d.threads << ")\n"
        << "  --chunk-size N         seeds per chunk (" << d.chunkSize << ")\n"
        << "  --checkpoint PATH      checkpoint file (" << d.checkpointPath << ")\n"
        << "  --out PATH             output JSONL file (" << d.outputPath << ")\n"
        << "  --no-resume            do not load checkpoint even if present\n"
        << "  --progress-secs N      progress print interval seconds (" << d.progressEverySeconds << ")\n"
        << "  --vertical-weight W    vertical distance weight (" << d.verticalWeight << ")\n"
        << "  --water-max N          max weighted distance start->water (" << d.waterMaxDist << ")\n"
        << "  --metal-max N          max weighted distance start->any metal volcano (" << d.metalMaxDist << ")\n"
        << "  --oil-max-pair N       max weighted pairwise distance oil reservoirs (" << d.oilMaxPairDist << ")\n"
        << "  --w-water W            score weight for water distance (" << d.wWater << ")\n"
        << "  --w-metal W            score weight for metal distance (" << d.wMetal << ")\n"
        << "  --w-oil W              score weight for oil clustering (" << d.wOil << ")\n"
        << "  --w-center W           score weight for center offset (" << d.wCenter << ")\n"
        << "  -h, --help             show this help\n";
}

static bool ParseSearchArgs(int argc, char **argv, SeedSearchConfig &cfg)
{
    bool customRequire = false;
    bool customForbid = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need = [&](std::string_view name) -> std::optional<std::string_view> {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return std::nullopt;
            }
            return std::string_view(argv[++i]);
        };

        if (a == "-h" || a == "--help") {
            PrintSearchUsage();
            return false;
        } else if (a == "--start") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n) {
                std::cerr << "Invalid --start: " << *v << "\n";
                return false;
            }
            cfg.seedStart = std::max(kSeedMin, *n);
        } else if (a == "--end") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n) {
                std::cerr << "Invalid --end: " << *v << "\n";
                return false;
            }
            cfg.seedEnd = std::min(kSeedMax, *n);
        } else if (a == "--mix") {
            auto v = need(a);
            if (!v)
                return false;
            cfg.mixingCode = std::string(*v);
        } else if (a == "--require-trait") {
            auto v = need(a);
            if (!v)
                return false;
            if (!customRequire) {
                cfg.requireTraits.clear();
                customRequire = true;
            }
            cfg.requireTraits.emplace_back(*v);
        } else if (a == "--forbid-trait") {
            auto v = need(a);
            if (!v)
                return false;
            if (!customForbid) {
                cfg.forbidTraits.clear();
                customForbid = true;
            }
            cfg.forbidTraits.emplace_back(*v);
        } else if (a == "--exact-traits") {
            cfg.exactTraits = true;
        } else if (a == "--threads") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n || *n <= 0) {
                std::cerr << "Invalid --threads: " << *v << "\n";
                return false;
            }
            cfg.threads = *n;
        } else if (a == "--chunk-size") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n || *n <= 0) {
                std::cerr << "Invalid --chunk-size: " << *v << "\n";
                return false;
            }
            cfg.chunkSize = *n;
        } else if (a == "--checkpoint") {
            auto v = need(a);
            if (!v)
                return false;
            cfg.checkpointPath = std::string(*v);
        } else if (a == "--out") {
            auto v = need(a);
            if (!v)
                return false;
            cfg.outputPath = std::string(*v);
        } else if (a == "--no-resume") {
            cfg.resume = false;
        } else if (a == "--progress-secs") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n || *n <= 0) {
                std::cerr << "Invalid --progress-secs: " << *v << "\n";
                return false;
            }
            cfg.progressEverySeconds = *n;
        } else if (a == "--vertical-weight") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseDouble(*v);
            if (!n || *n <= 0.0) {
                std::cerr << "Invalid --vertical-weight: " << *v << "\n";
                return false;
            }
            cfg.verticalWeight = *n;
        } else if (a == "--water-max") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n || *n < 0) {
                std::cerr << "Invalid --water-max: " << *v << "\n";
                return false;
            }
            cfg.waterMaxDist = *n;
        } else if (a == "--metal-max") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n || *n < 0) {
                std::cerr << "Invalid --metal-max: " << *v << "\n";
                return false;
            }
            cfg.metalMaxDist = *n;
        } else if (a == "--oil-max-pair") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseInt32(*v);
            if (!n || *n < 0) {
                std::cerr << "Invalid --oil-max-pair: " << *v << "\n";
                return false;
            }
            cfg.oilMaxPairDist = *n;
        } else if (a == "--w-water") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseDouble(*v);
            if (!n) {
                std::cerr << "Invalid --w-water: " << *v << "\n";
                return false;
            }
            cfg.wWater = *n;
        } else if (a == "--w-metal") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseDouble(*v);
            if (!n) {
                std::cerr << "Invalid --w-metal: " << *v << "\n";
                return false;
            }
            cfg.wMetal = *n;
        } else if (a == "--w-oil") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseDouble(*v);
            if (!n) {
                std::cerr << "Invalid --w-oil: " << *v << "\n";
                return false;
            }
            cfg.wOil = *n;
        } else if (a == "--w-center") {
            auto v = need(a);
            if (!v)
                return false;
            auto n = ParseDouble(*v);
            if (!n) {
                std::cerr << "Invalid --w-center: " << *v << "\n";
                return false;
            }
            cfg.wCenter = *n;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            PrintSearchUsage();
            return false;
        }
    }
    if (cfg.seedStart > cfg.seedEnd) {
        std::cerr << "--start must be <= --end\n";
        return false;
    }
    if (cfg.chunkSize <= 0) {
        std::cerr << "--chunk-size must be > 0\n";
        return false;
    }
    if (cfg.requireTraits.empty()) {
        std::cerr << "At least one --require-trait is required (or omit to use defaults)\n";
        return false;
    }
    return true;
}

static std::optional<int32_t> TryLoadNextSeedFromCheckpoint(const SeedSearchConfig &cfg)
{
    if (!cfg.resume) {
        return std::nullopt;
    }
    auto content = ReadFileToString(cfg.checkpointPath);
    if (!content) {
        return std::nullopt;
    }
    // Tiny JSON "parser": just extract "next_seed":<int>.
    auto pos = content->find("\"next_seed\"");
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = content->find(':', pos);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    ++pos;
    while (pos < content->size() && std::isspace((unsigned char)(*content)[pos])) {
        ++pos;
    }
    size_t end = pos;
    while (end < content->size() && (std::isdigit((unsigned char)(*content)[end]) || (*content)[end] == '-')) {
        ++end;
    }
    auto n = ParseInt32(std::string_view(*content).substr(pos, end - pos));
    if (!n) {
        return std::nullopt;
    }
    return *n;
}

static void WriteResultJsonl(std::ofstream &out, const SeedEval &e)
{
    auto arrVec2 = [](const std::vector<Vector2i> &v) {
        std::ostringstream s;
        s << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i)
                s << ",";
            s << "{\"x\":" << v[i].x << ",\"y\":" << v[i].y << "}";
        }
        s << "]";
        return s.str();
    };
    auto arrStr = [](const std::vector<std::string> &v) {
        std::ostringstream s;
        s << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i)
                s << ",";
            s << "\"" << JsonEscape(v[i]) << "\"";
        }
        s << "]";
        return s.str();
    };

    std::ostringstream s;
    s << "{";
    s << "\"seed\":" << e.seed << ",";
    s << "\"code\":\"" << JsonEscape(e.code) << "\",";
    s << "\"score\":" << std::fixed << std::setprecision(2) << e.score << ",";
    s << "\"world_size\":{\"x\":" << e.worldSize.x << ",\"y\":" << e.worldSize.y << "},";
    s << "\"start\":{\"x\":" << e.start.x << ",\"y\":" << e.start.y << "},";
    s << "\"center_offset_x\":" << e.centerOffsetX << ",";
    s << "\"dist_water\":" << e.distWater << ",";
    s << "\"dist_metal_any\":" << e.distMetalAny << ",";
    s << "\"oil_max_pair_dist\":" << e.oilMaxPairDist << ",";
    s << "\"traits\":" << arrStr(e.traits) << ",";
    s << "\"geysers\":{";
    s << "\"water\":" << arrVec2(e.water) << ",";
    s << "\"copper\":" << arrVec2(e.copper) << ",";
    s << "\"iron\":" << arrVec2(e.iron) << ",";
    s << "\"gold\":" << arrVec2(e.gold) << ",";
    s << "\"oil_reservoirs\":" << arrVec2(e.oilReservoirs);
    s << "}";
    s << "}\n";
    out << s.str();
}

} // namespace

int SeedSearchMain(int argc, char **argv)
{
    signal(SIGINT, OnSignal);
    signal(SIGTERM, OnSignal);

    SeedSearchConfig cfg;
    cfg.threads = (int)std::max(1u, std::thread::hardware_concurrency());
    if (!ParseSearchArgs(argc, argv, cfg)) {
        // Help or error already printed.
        return 2;
    }

    // Load checkpoint if present.
    int32_t effectiveStart = cfg.seedStart;
    if (auto resumeSeed = TryLoadNextSeedFromCheckpoint(cfg)) {
        if (*resumeSeed > effectiveStart && *resumeSeed <= cfg.seedEnd) {
            effectiveStart = *resumeSeed;
            std::cerr << "Resuming from checkpoint next_seed=" << effectiveStart
                      << " (" << cfg.checkpointPath << ")\n";
        }
    }

    // Load zipped settings once; each thread parses into its own SettingsCache.
    auto zip = ReadFileToString(SETTING_ASSET_FILEPATH);
    if (!zip) {
        std::cerr << "Failed to read settings zip: " << SETTING_ASSET_FILEPATH
                  << "\n";
        return 1;
    }

    // Output file: append so resume doesn't clobber.
    std::ofstream out(cfg.outputPath, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Failed to open output: " << cfg.outputPath << "\n";
        return 1;
    }

    std::mutex outMutex;
    std::atomic<uint64_t> seedsTried{0};
    std::atomic<uint64_t> seedsMatched{0};

    WorkQueue work;
    CompletionQueue done;

    // Limit outstanding chunks so checkpoint can move forward quickly.
    const size_t maxOutstanding = (size_t)cfg.threads * 4;
    std::atomic<size_t> outstanding{0};

    // Chunk id is relative to cfg.seedStart (not effectiveStart) for stability.
    auto chunkIdForSeedBegin = [&](int32_t seedBegin) -> uint64_t {
        return (uint64_t)((seedBegin - cfg.seedStart) / cfg.chunkSize);
    };

    std::atomic<int32_t> nextSeedToEnqueue{effectiveStart};

    auto enqueueMore = [&]() {
        while (!g_stop.load(std::memory_order_relaxed) &&
               outstanding.load(std::memory_order_relaxed) < maxOutstanding) {
            int32_t seedBegin = nextSeedToEnqueue.load(std::memory_order_relaxed);
            if (seedBegin > cfg.seedEnd) {
                break;
            }
            int64_t endEx = (int64_t)seedBegin + (int64_t)cfg.chunkSize;
            if (endEx > (int64_t)cfg.seedEnd + 1) {
                endEx = (int64_t)cfg.seedEnd + 1;
            }
            WorkItem item;
            item.seedBegin = seedBegin;
            item.seedEndExclusive = (int32_t)endEx;
            item.chunkId = chunkIdForSeedBegin(seedBegin);
            nextSeedToEnqueue.store(item.seedEndExclusive, std::memory_order_relaxed);
            outstanding.fetch_add(1, std::memory_order_relaxed);
            work.Push(item);
        }
    };

    // Workers
    std::vector<std::thread> threads;
    threads.reserve((size_t)cfg.threads);
    for (int t = 0; t < cfg.threads; ++t) {
        threads.emplace_back([&, t] {
            ThreadState st;
            std::string error;
            if (!InitThreadState(st, *zip, cfg, error)) {
                std::lock_guard<std::mutex> lock(outMutex);
                std::cerr << "Worker init failed: " << error << "\n";
                g_stop.store(true, std::memory_order_relaxed);
                return;
            }
            while (true) {
                auto itemOpt = work.Pop();
                if (!itemOpt) {
                    return;
                }
                auto item = *itemOpt;
                bool completed = true;
                for (int32_t seed = item.seedBegin; seed < item.seedEndExclusive; ++seed) {
                    if (g_stop.load(std::memory_order_relaxed)) {
                        completed = false;
                        break;
                    }
                    seedsTried.fetch_add(1, std::memory_order_relaxed);
                    auto eval = EvaluateSeed(st, seed, cfg);
                    if (!eval) {
                        continue;
                    }
                    seedsMatched.fetch_add(1, std::memory_order_relaxed);
                    {
                        std::lock_guard<std::mutex> lock(outMutex);
                        WriteResultJsonl(out, *eval);
                        out.flush();
                        std::cout << "match seed=" << eval->seed
                                  << " score=" << std::fixed << std::setprecision(2)
                                  << eval->score << " code=" << eval->code
                                  << " water_d=" << eval->distWater
                                  << " metal_d=" << eval->distMetalAny
                                  << " oil_max_pair_d=" << eval->oilMaxPairDist
                                  << " start=(" << eval->start.x << "," << eval->start.y
                                  << ")\n";
                    }
                }
                if (completed) {
                    done.Push(CompletedChunk{item.chunkId});
                } else {
                    // Not completed: drop this chunk so checkpoint won't advance past it.
                    // The caller can resume and re-check this range.
                    done.Push(CompletedChunk{UINT64_MAX});
                    return;
                }
            }
        });
    }

    // Progress printer.
    std::thread progress([&] {
        using clock = std::chrono::steady_clock;
        auto start = clock::now();
        auto last = start;
        uint64_t lastCount = 0;
        while (!g_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(cfg.progressEverySeconds));
            auto now = clock::now();
            uint64_t count = seedsTried.load(std::memory_order_relaxed);
            double dt = std::chrono::duration<double>(now - last).count();
            double rate = dt > 0 ? (double)(count - lastCount) / dt : 0.0;
            double totalDt = std::chrono::duration<double>(now - start).count();
            double avgRate = totalDt > 0 ? (double)count / totalDt : 0.0;
            std::cerr << "progress tried=" << count
                      << " matched=" << seedsMatched.load(std::memory_order_relaxed)
                      << " rate=" << std::fixed << std::setprecision(0) << rate
                      << "/s avg=" << avgRate << "/s"
                      << " next_seed=" << nextSeedToEnqueue.load(std::memory_order_relaxed)
                      << "\n";
            last = now;
            lastCount = count;
        }
    });

    // Coordinator: enqueue initial work and keep checkpointing as chunks complete.
    enqueueMore();

    uint64_t nextCheckpointChunkId = chunkIdForSeedBegin(effectiveStart);
    std::set<uint64_t> completed;

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    while (true) {
        if (g_stop.load(std::memory_order_relaxed)) {
            break;
        }
        auto msg = done.Pop();
        if (!msg) {
            break;
        }
        if (msg->chunkId == UINT64_MAX) {
            // A worker stopped mid-chunk; stop coordinating new work.
            g_stop.store(true, std::memory_order_relaxed);
            break;
        }
        completed.insert(msg->chunkId);
        outstanding.fetch_sub(1, std::memory_order_relaxed);

        // Advance checkpoint over contiguous completed chunks.
        while (completed.contains(nextCheckpointChunkId)) {
            completed.erase(nextCheckpointChunkId);
            ++nextCheckpointChunkId;
            int64_t nextSeed64 =
                (int64_t)cfg.seedStart + (int64_t)nextCheckpointChunkId * (int64_t)cfg.chunkSize;
            int32_t nextSeed = (nextSeed64 > (int64_t)cfg.seedEnd + 1)
                                   ? (cfg.seedEnd + 1)
                                   : (int32_t)nextSeed64;
            auto now = clock::now();
            double dt = std::chrono::duration<double>(now - t0).count();
            double avg = dt > 0 ? (double)seedsTried.load(std::memory_order_relaxed) / dt : 0.0;
            auto json = ToCheckpointJson(cfg, nextSeed,
                                         seedsTried.load(std::memory_order_relaxed),
                                         seedsMatched.load(std::memory_order_relaxed),
                                         avg, nextCheckpointChunkId);
            WriteFileAtomic(cfg.checkpointPath, json);
        }

        enqueueMore();

        if (nextSeedToEnqueue.load(std::memory_order_relaxed) > cfg.seedEnd &&
            outstanding.load(std::memory_order_relaxed) == 0) {
            break;
        }
    }

    // Shut down.
    g_stop.store(true, std::memory_order_relaxed);
    work.Close();
    done.Close();
    if (progress.joinable()) {
        progress.join();
    }
    for (auto &th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Final checkpoint write.
    {
        auto now = clock::now();
        double dt = std::chrono::duration<double>(now - t0).count();
        double avg = dt > 0 ? (double)seedsTried.load(std::memory_order_relaxed) / dt : 0.0;
        int64_t nextSeed64 =
            (int64_t)cfg.seedStart + (int64_t)nextCheckpointChunkId * (int64_t)cfg.chunkSize;
        int32_t nextSeed =
            (nextSeed64 > (int64_t)cfg.seedEnd + 1) ? (cfg.seedEnd + 1) : (int32_t)nextSeed64;
        auto json = ToCheckpointJson(cfg, nextSeed,
                                     seedsTried.load(std::memory_order_relaxed),
                                     seedsMatched.load(std::memory_order_relaxed),
                                     avg, nextCheckpointChunkId);
        WriteFileAtomic(cfg.checkpointPath, json);
    }

    std::cerr << "done tried=" << seedsTried.load() << " matched=" << seedsMatched.load()
              << " stop=" << (g_stop.load() ? "true" : "false") << "\n";
    return 0;
}

#endif // EMSCRIPTEN
