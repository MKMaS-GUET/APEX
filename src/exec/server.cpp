#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exec/apex.hpp"
#include "httplib.h"
#include "index/index_builder.hpp"
#include "index/index_retriever.hpp"
#include "query/query_executor.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace fs = std::filesystem;

namespace {
constexpr std::string_view kDbArchiveDir = "./DB_DATA_ARCHIVE";
constexpr std::string_view kUiDir = "./src/exec/web";
constexpr std::string_view kDefaultHost = "0.0.0.0";
constexpr int kDefaultPort = 8080;
constexpr uint kDefaultQueryThreads = 32;
constexpr uint kDefaultMaxUploadMB = 20480;  // 20GB
constexpr uint64_t kNtValidationSampleLines = 100000;
constexpr uint64_t kMaxQueryResultRows = 1000;
constexpr std::string_view kServerHelpInfo =
    "Usage: apex server [OPTIONS]\n"
    "\n"
    "Options:\n"
    "  --host <HOST>            Bind host (default: 0.0.0.0).\n"
    "  --port <PORT>            Bind port (default: 8080).\n"
    "  --threads <NUM>          Query threads for API requests (default: 32).\n"
    "  --max-upload-mb <NUM>    Max upload payload in MB (default: 20480).\n"
    "  -h, --help               Show this help message and exit.\n";

struct DatabaseStats {
    uint64_t subject_count = 0;
    uint64_t predicate_count = 0;
    uint64_t object_count = 0;
    uint64_t shared_count = 0;
    uint64_t entity_count = 0;
    uint64_t triple_count = 0;
    bool triple_count_exact = false;
};

struct DatabaseDiskUsage {
    uint64_t index_size_bytes = 0;
    uint64_t dictionary_size_bytes = 0;
    uint64_t total_size_bytes = 0;
};

struct ActiveDatabaseSnapshot {
    bool exists = false;
    std::string name;
    std::shared_ptr<IndexRetriever> index;
    DatabaseStats stats;
};

fs::path ResolveUiDir() {
    std::vector<fs::path> candidates;
    candidates.emplace_back(fs::path(kUiDir));

    const fs::path cwd = fs::current_path();
    candidates.emplace_back(cwd / "src/exec/web");
    candidates.emplace_back(cwd / "../src/exec/web");

#ifdef __linux__
    std::error_code ec;
    const fs::path exe_path = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && !exe_path.empty()) {
        const fs::path exe_dir = exe_path.parent_path();
        candidates.emplace_back(exe_dir / "../src/exec/web");
        candidates.emplace_back(exe_dir / "../../src/exec/web");
    }
#endif

    for (const auto& candidate : candidates) {
        std::error_code ec;
        const fs::path canonical_path = fs::weakly_canonical(candidate, ec);
        if (!ec && fs::exists(canonical_path) && fs::is_directory(canonical_path))
            return canonical_path;
    }
    return fs::absolute(candidates.front());
}

std::string GetArgValue(int argc, char** argv, const std::string& key, const std::string& default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key)
            return argv[i + 1];
    }
    return default_value;
}

uint GetArgUInt(int argc, char** argv, const std::string& key, uint default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key)
            return static_cast<uint>(std::stoul(argv[i + 1]));
    }
    return default_value;
}

bool HasArg(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == key)
            return true;
    }
    return false;
}

void SetCorsHeaders(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

void WriteError(httplib::Response& res, int status, const std::string& message) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();
    writer.Key("ok");
    writer.Bool(false);
    writer.Key("message");
    writer.String(message.c_str());
    writer.EndObject();

    res.status = status;
    SetCorsHeaders(res);
    res.set_content(buf.GetString(), "application/json;charset=utf-8");
}

bool ParseJsonObject(const httplib::Request& req, rapidjson::Document& doc, std::string& err) {
    if (req.body.empty())
        return false;
    doc.Parse(req.body.c_str());
    if (doc.HasParseError()) {
        err = "Request body is not valid JSON.";
        return false;
    }
    if (!doc.IsObject()) {
        err = "JSON body must be an object.";
        return false;
    }
    return true;
}

std::string ReadStringField(const httplib::Request& req,
                            const rapidjson::Document* doc,
                            std::initializer_list<const char*> keys) {
    for (const auto* key : keys) {
        if (req.has_param(key))
            return req.get_param_value(key);
    }
    if (doc != nullptr) {
        for (const auto* key : keys) {
            if (doc->HasMember(key) && (*doc)[key].IsString())
                return (*doc)[key].GetString();
        }
    }
    return {};
}

DatabaseStats BuildDatabaseStats(const std::shared_ptr<IndexRetriever>& index) {
    DatabaseStats stats;
    stats.subject_count = index->subject_cnt();
    stats.predicate_count = index->predicate_cnt();
    stats.object_count = index->object_cnt();
    stats.shared_count = index->shared_cnt();
    stats.entity_count = stats.subject_count + stats.object_count + stats.shared_count;
    stats.triple_count = index->triple_cnt();
    stats.triple_count_exact = (stats.triple_count != 0);
    return stats;
}

bool ReadDatabaseStatsFromMetadata(const std::string& db_name, DatabaseStats* out_stats) {
    if (out_stats == nullptr)
        return false;

    const fs::path metadata_path = fs::path(kDbArchiveDir) / db_name / "dictionary" / "menagement_data";
    if (!fs::exists(metadata_path))
        return false;

    std::ifstream in(metadata_path, std::ios::binary | std::ios::ate);
    if (!in.is_open())
        return false;

    const std::streamsize bytes = in.tellg();
    if (bytes < static_cast<std::streamsize>(4 * sizeof(uint64_t)))
        return false;

    const size_t count = static_cast<size_t>(bytes / static_cast<std::streamsize>(sizeof(uint64_t)));
    std::vector<uint64_t> meta(count, 0);
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(meta.data()), static_cast<std::streamsize>(count * sizeof(uint64_t)));
    if (!in)
        return false;

    DatabaseStats stats;
    stats.subject_count = meta[0];
    stats.predicate_count = meta[1];
    stats.object_count = meta[2];
    stats.shared_count = meta[3];
    stats.entity_count = stats.subject_count + stats.object_count + stats.shared_count;
    if (count > 7) {
        stats.triple_count = meta[7];
        stats.triple_count_exact = true;
    }
    *out_stats = stats;
    return true;
}

uint64_t DirectorySizeBytes(const fs::path& root) {
    std::error_code ec;
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec))
        return 0;

    uint64_t total = 0;
    fs::recursive_directory_iterator it(
        root, fs::directory_options::skip_permission_denied, ec);
    fs::recursive_directory_iterator end;
    while (!ec && it != end) {
        std::error_code file_ec;
        if (it->is_regular_file(file_ec)) {
            const uint64_t file_size = it->file_size(file_ec);
            if (!file_ec)
                total += file_size;
        }
        it.increment(ec);
    }
    return total;
}

DatabaseDiskUsage ReadDatabaseDiskUsage(const std::string& db_name) {
    const fs::path db_path = fs::path(kDbArchiveDir) / db_name;
    DatabaseDiskUsage usage;
    usage.index_size_bytes = DirectorySizeBytes(db_path / "index");
    usage.dictionary_size_bytes = DirectorySizeBytes(db_path / "dictionary");
    usage.total_size_bytes = usage.index_size_bytes + usage.dictionary_size_bytes;
    return usage;
}

std::string SanitizeFilename(const std::string& filename) {
    if (filename.empty())
        return "upload_data_file";

    std::string out;
    out.reserve(filename.size());
    for (char ch : filename) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) || ch == '.' || ch == '_' || ch == '-')
            out.push_back(ch);
        else
            out.push_back('_');
    }
    return out;
}

bool SaveUploadedDataFile(const std::string& db_name,
                          const httplib::FormData& form_data,
                          std::string& saved_path,
                          std::string& err) {
    if (form_data.content.empty()) {
        err = "Uploaded file is empty.";
        return false;
    }

    const fs::path upload_dir = fs::path(kDbArchiveDir) / ".uploads";
    std::error_code ec;
    fs::create_directories(upload_dir, ec);
    if (ec) {
        err = "Failed to create upload temp directory: " + upload_dir.string();
        return false;
    }

    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    const std::string safe_name = SanitizeFilename(form_data.filename);
    const fs::path temp_path = upload_dir / (db_name + "_" + std::to_string(stamp) + "_" + safe_name);

    std::ofstream out(temp_path, std::ios::binary);
    if (!out.is_open()) {
        err = "Failed to open temp file for upload.";
        return false;
    }
    out.write(form_data.content.data(), static_cast<std::streamsize>(form_data.content.size()));
    if (!out) {
        err = "Failed to write uploaded file.";
        return false;
    }
    out.close();

    saved_path = temp_path.string();
    return true;
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string Trim(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])))
        ++begin;

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;

    return value.substr(begin, end - begin);
}

bool HasNtExtension(const std::string& filename) {
    if (filename.empty())
        return false;
    return ToLower(fs::path(filename).extension().string()) == ".nt";
}

bool ParseUInt64(const std::string& value, uint64_t* out) {
    if (out == nullptr || value.empty())
        return false;
    try {
        size_t parsed = 0;
        const uint64_t v = std::stoull(value, &parsed, 10);
        if (parsed != value.size())
            return false;
        *out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool ValidateNtFile(const std::string& file_path, std::string& err, uint64_t* out_checked_lines = nullptr) {
    if (!HasNtExtension(file_path)) {
        err = "Only .nt files are supported.";
        return false;
    }

    std::ifstream fin(file_path, std::ios::in);
    if (!fin.is_open()) {
        err = "Failed to open uploaded file: " + file_path;
        return false;
    }

    uint64_t line_no = 0;
    uint64_t checked_lines = 0;
    uint64_t valid_triple_lines = 0;
    std::string line;
    while (std::getline(fin, line)) {
        ++line_no;
        std::string trimmed = Trim(line);
        if (trimmed.empty())
            continue;
        if (trimmed.front() == '#')
            continue;

        ++checked_lines;

        if (trimmed.back() != '.') {
            err = "File is not valid .nt format. Line " + std::to_string(line_no) + " must end with '.'.";
            return false;
        }

        std::istringstream iss(trimmed);
        std::string subject;
        std::string predicate;
        if (!(iss >> subject >> predicate)) {
            err = "File is not valid .nt format. Line " + std::to_string(line_no) + " is incomplete.";
            return false;
        }

        std::string object_and_dot;
        std::getline(iss, object_and_dot);
        object_and_dot = Trim(object_and_dot);
        if (object_and_dot.empty()) {
            err = "File is not valid .nt format. Line " + std::to_string(line_no) + " is missing object.";
            return false;
        }

        while (!object_and_dot.empty() && object_and_dot.back() == '.')
            object_and_dot.pop_back();
        object_and_dot = Trim(object_and_dot);
        if (object_and_dot.empty()) {
            err = "File is not valid .nt format. Line " + std::to_string(line_no) + " is missing object.";
            return false;
        }

        ++valid_triple_lines;
        if (checked_lines >= kNtValidationSampleLines)
            break;
    }

    if (valid_triple_lines == 0) {
        err = "Uploaded .nt file contains no triples.";
        return false;
    }

    if (out_checked_lines != nullptr)
        *out_checked_lines = checked_lines;
    return true;
}

class UploadRegistry {
   public:
    bool InitSession(const std::string& db_name,
                     const std::string& filename,
                     std::string& upload_id,
                     std::string& err) {
        if (db_name.empty()) {
            err = "Database name is required.";
            return false;
        }
        if (!HasNtExtension(filename)) {
            err = "Only .nt files can be uploaded.";
            return false;
        }

        const fs::path upload_dir = fs::path(kDbArchiveDir) / ".uploads";
        std::error_code ec;
        fs::create_directories(upload_dir, ec);
        if (ec) {
            err = "Failed to create upload temp directory: " + upload_dir.string();
            return false;
        }

        const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
        const std::string safe_name = SanitizeFilename(db_name);
        upload_id = safe_name + "_" + std::to_string(stamp);
        const fs::path temp_path = upload_dir / (upload_id + ".nt");

        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            err = "Failed to initialize upload temp file.";
            return false;
        }
        out.close();

        std::scoped_lock lock(mu_);
        sessions_[upload_id] = UploadSession{db_name, temp_path.string(), 0, 0};
        return true;
    }

    bool AppendChunk(const std::string& upload_id,
                     uint64_t chunk_index,
                     const std::string& chunk_content,
                     uint64_t* out_total_bytes,
                     std::string& err) {
        if (upload_id.empty()) {
            err = "upload_id is required.";
            return false;
        }
        if (chunk_content.empty()) {
            err = "Chunk content is empty.";
            return false;
        }

        std::scoped_lock lock(mu_);
        auto it = sessions_.find(upload_id);
        if (it == sessions_.end()) {
            err = "Upload session not found: " + upload_id;
            return false;
        }
        UploadSession& session = it->second;
        if (chunk_index != session.next_chunk_index) {
            err = "Chunk index mismatch. Expected " + std::to_string(session.next_chunk_index) +
                  ", got " + std::to_string(chunk_index) + ".";
            return false;
        }

        std::ofstream out(session.temp_file_path, std::ios::binary | std::ios::app);
        if (!out.is_open()) {
            err = "Failed to open temp upload file for appending.";
            return false;
        }
        out.write(chunk_content.data(), static_cast<std::streamsize>(chunk_content.size()));
        if (!out) {
            err = "Failed to write upload chunk.";
            return false;
        }
        out.close();

        session.received_bytes += static_cast<uint64_t>(chunk_content.size());
        session.next_chunk_index++;
        if (out_total_bytes != nullptr)
            *out_total_bytes = session.received_bytes;
        return true;
    }

    bool ResolveFilePath(const std::string& upload_id, std::string& file_path, std::string& err) const {
        if (upload_id.empty()) {
            err = "upload_id is required.";
            return false;
        }

        std::scoped_lock lock(mu_);
        auto it = sessions_.find(upload_id);
        if (it == sessions_.end()) {
            err = "Upload session not found: " + upload_id;
            return false;
        }
        file_path = it->second.temp_file_path;
        return true;
    }

    void RemoveSession(const std::string& upload_id) {
        if (upload_id.empty())
            return;

        std::string file_path;
        {
            std::scoped_lock lock(mu_);
            auto it = sessions_.find(upload_id);
            if (it == sessions_.end())
                return;
            file_path = it->second.temp_file_path;
            sessions_.erase(it);
        }

        std::error_code ec;
        fs::remove(file_path, ec);
    }

   private:
    struct UploadSession {
        std::string database;
        std::string temp_file_path;
        uint64_t next_chunk_index = 0;
        uint64_t received_bytes = 0;
    };

    mutable std::mutex mu_;
    std::unordered_map<std::string, UploadSession> sessions_;
};

class CreateProgressTracker {
   public:
    struct Snapshot {
        bool running = false;
        bool cancel_requested = false;
        int progress_percent = 0;
        std::string status = "idle";
        std::string database;
        std::string step;
        std::string message;
    };

    bool TryStart(const std::string& database, std::string& err) {
        std::scoped_lock lock(mu_);
        if (state_.running) {
            err = "A database creation task is already running.";
            return false;
        }

        state_.running = true;
        state_.cancel_requested = false;
        state_.progress_percent = 0;
        state_.status = "running";
        state_.database = database;
        state_.step = "准备创建";
        state_.message = "任务已开始。";
        return true;
    }

    void Update(int progress_percent, const std::string& step, const std::string& message) {
        std::scoped_lock lock(mu_);
        if (!state_.running)
            return;
        state_.progress_percent = std::max(0, std::min(99, progress_percent));
        state_.step = step;
        state_.message = message;
    }

    void FinishSuccess(const std::string& message) {
        std::scoped_lock lock(mu_);
        state_.running = false;
        state_.cancel_requested = false;
        state_.progress_percent = 100;
        state_.status = "success";
        state_.step = "创建完成";
        state_.message = message;
    }

    void FinishError(const std::string& message) {
        std::scoped_lock lock(mu_);
        state_.running = false;
        state_.cancel_requested = false;
        state_.status = "error";
        state_.step = "创建失败";
        state_.message = message;
    }

    void FinishCancelled(const std::string& message) {
        std::scoped_lock lock(mu_);
        state_.running = false;
        state_.cancel_requested = false;
        state_.status = "cancelled";
        state_.step = "已取消";
        state_.message = message;
    }

    void RequestCancel() {
        std::scoped_lock lock(mu_);
        if (!state_.running)
            return;
        state_.cancel_requested = true;
        state_.message = "已收到取消请求，等待当前步骤结束。";
    }

    bool IsRunning() const {
        std::scoped_lock lock(mu_);
        return state_.running;
    }

    bool IsCancelRequested() const {
        std::scoped_lock lock(mu_);
        return state_.cancel_requested;
    }

    Snapshot GetSnapshot() const {
        std::scoped_lock lock(mu_);
        return state_;
    }

   private:
    mutable std::mutex mu_;
    Snapshot state_;
};

template <typename Writer>
void WriteDatabaseStats(Writer& writer, const DatabaseStats& stats) {
    writer.StartObject();
    writer.Key("subject_count");
    writer.Uint64(stats.subject_count);
    writer.Key("predicate_count");
    writer.Uint64(stats.predicate_count);
    writer.Key("object_count");
    writer.Uint64(stats.object_count);
    writer.Key("shared_count");
    writer.Uint64(stats.shared_count);
    writer.Key("entity_count");
    writer.Uint64(stats.entity_count);
    writer.Key("triple_count");
    writer.Uint64(stats.triple_count);
    writer.Key("triple_count_exact");
    writer.Bool(stats.triple_count_exact);
    writer.EndObject();
}

template <typename Writer>
void WriteDatabaseList(Writer& writer, const std::vector<std::string>& names) {
    writer.StartArray();
    for (const auto& name : names)
        writer.String(name.c_str());
    writer.EndArray();
}

class DatabaseManager {
   public:
    std::vector<std::string> ExistingDatabases() const {
        std::vector<std::string> dbs;
        const fs::path root(kDbArchiveDir);
        if (!fs::exists(root) || !fs::is_directory(root))
            return dbs;

        for (const auto& entry : fs::directory_iterator(root)) {
            if (!entry.is_directory())
                continue;
            const fs::path db_path = entry.path();
            if (fs::exists(db_path / "index") && fs::exists(db_path / "dictionary"))
                dbs.push_back(db_path.filename().string());
        }
        std::sort(dbs.begin(), dbs.end());
        return dbs;
    }

    std::vector<std::string> LoadedDatabases() const {
        std::scoped_lock lock(mu_);
        if (active_name_.empty())
            return {};
        return {active_name_};
    }

    std::vector<std::pair<std::string, DatabaseStats>> LoadedDatabaseDetails() const {
        std::scoped_lock lock(mu_);
        if (active_name_.empty())
            return {};
        return {{active_name_, active_stats_}};
    }

    std::vector<std::pair<std::string, DatabaseStats>> ExistingDatabaseDetails() const {
        std::vector<std::pair<std::string, DatabaseStats>> details;
        const auto names = ExistingDatabases();
        details.reserve(names.size());
        for (const auto& name : names) {
            DatabaseStats stats;
            ReadDatabaseStatsFromMetadata(name, &stats);
            details.push_back({name, stats});
        }
        return details;
    }

    ActiveDatabaseSnapshot Active() const {
        std::scoped_lock lock(mu_);
        ActiveDatabaseSnapshot snapshot;
        if (active_name_.empty())
            return snapshot;
        snapshot.exists = true;
        snapshot.name = active_name_;
        snapshot.index = active_index_;
        snapshot.stats = active_stats_;
        return snapshot;
    }

    bool CreateDatabase(const std::string& name, const std::string& data_file, std::string& err) {
        if (name.empty()) {
            err = "Database name is required.";
            return false;
        }
        if (data_file.empty()) {
            err = "Data file path is required.";
            return false;
        }
        if (!fs::exists(data_file)) {
            err = "Data file does not exist: " + data_file;
            return false;
        }

        const fs::path db_path = DbPath(name);
        if (fs::exists(db_path / "index") && fs::exists(db_path / "dictionary")) {
            err = "Database already exists: " + name;
            return false;
        }

        try {
            IndexBuilder builder(name, data_file);
            if (!builder.Build()) {
                err = "Failed to build database: " + name;
                return false;
            }
        } catch (const std::exception& e) {
            err = std::string("Create database failed: ") + e.what();
            return false;
        }
        return true;
    }

    bool RemoveDatabaseFiles(const std::string& name, std::string& err) {
        if (name.empty()) {
            err = "Database name is required.";
            return false;
        }

        {
            std::scoped_lock lock(mu_);
            if (active_name_ == name && active_index_) {
                active_index_->Close();
                active_index_.reset();
                active_name_.clear();
                active_stats_ = DatabaseStats{};
            }
        }

        const fs::path db_path = DbPath(name);
        if (!fs::exists(db_path))
            return true;

        std::error_code ec;
        fs::remove_all(db_path, ec);
        if (ec) {
            err = "Failed to remove database files: " + db_path.string();
            return false;
        }
        return true;
    }

    bool LoadDatabase(const std::string& name, std::string& err, DatabaseStats* out_stats = nullptr) {
        if (name.empty()) {
            err = "Database name is required.";
            return false;
        }

        const fs::path db_path = DbPath(name);
        if (!fs::exists(db_path / "index") || !fs::exists(db_path / "dictionary")) {
            err = "Database not found: " + name;
            return false;
        }

        std::shared_ptr<IndexRetriever> index;
        try {
            // Server needs ID -> string conversion for JSON results, so dictionary nodes must be loaded.
            index = std::make_shared<IndexRetriever>(db_path.string(), true);
        } catch (const std::exception& e) {
            err = std::string("Load database failed: ") + e.what();
            return false;
        }
        const DatabaseStats stats = BuildDatabaseStats(index);

        std::scoped_lock lock(mu_);
        // Merge load & switch behavior:
        // If there is an active database, release its resources before activating the new one.
        if (active_index_) {
            active_index_->Close();
            active_index_.reset();
            active_name_.clear();
        }
        active_name_ = name;
        active_index_ = index;
        active_stats_ = stats;

        if (out_stats != nullptr)
            *out_stats = stats;
        return true;
    }

   private:
    static fs::path DbPath(const std::string& db_name) { return fs::path(kDbArchiveDir) / db_name; }

    mutable std::mutex mu_;
    std::string active_name_;
    std::shared_ptr<IndexRetriever> active_index_;
    DatabaseStats active_stats_;
};
}  // namespace

namespace apex {

int Server(int argc, char** argv) {
    if (HasArg(argc, argv, "-h") || HasArg(argc, argv, "--help")) {
        std::cout << kServerHelpInfo << std::endl;
        return 0;
    }

    const std::string host = GetArgValue(argc, argv, "--host", std::string(kDefaultHost));
    const int port = static_cast<int>(GetArgUInt(argc, argv, "--port", kDefaultPort));
    const uint max_threads = GetArgUInt(argc, argv, "--threads", kDefaultQueryThreads);
    const uint max_upload_mb = GetArgUInt(argc, argv, "--max-upload-mb", kDefaultMaxUploadMB);

    DatabaseManager manager;
    UploadRegistry upload_registry;
    CreateProgressTracker create_tracker;
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(8); };
    server.set_payload_max_length(static_cast<size_t>(max_upload_mb) * 1024 * 1024);

    const auto reject_if_creating = [&](httplib::Response& res) -> bool {
        if (!create_tracker.IsRunning())
            return false;
        WriteError(res, 423, "Database creation is in progress. Only cancel is allowed.");
        return true;
    };

    const fs::path ui_dir = ResolveUiDir();
    if (!server.set_mount_point("/apex/ui", ui_dir.string())) {
        std::cerr << "warning: failed to mount frontend directory at " << ui_dir
                  << ", cwd=" << fs::current_path() << std::endl;
    } else {
        std::cout << "Mounted UI directory: " << ui_dir << std::endl;
    }

    const auto options_handler = [](const httplib::Request&, httplib::Response& res) {
        SetCorsHeaders(res);
        res.status = 204;
    };

    server.Get("/", [](const httplib::Request&, httplib::Response& res) { res.set_redirect("/apex/ui/"); });
    server.Get("/apex", [](const httplib::Request&, httplib::Response& res) { res.set_redirect("/apex/ui/"); });
    server.Get("/apex/ui", [](const httplib::Request&, httplib::Response& res) { res.set_redirect("/apex/ui/"); });

    server.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        SetCorsHeaders(res);
        res.set_content("{\"status\":\"ok\"}", "application/json;charset=utf-8");
    });

    server.Post("/apex/databases/upload/init", [&](const httplib::Request& req, httplib::Response& res) {
        if (create_tracker.IsRunning()) {
            WriteError(res, 423, "Database creation is already running.");
            return;
        }

        rapidjson::Document doc;
        std::string parse_err;
        const rapidjson::Document* doc_ptr = nullptr;
        if (!req.body.empty()) {
            if (!ParseJsonObject(req, doc, parse_err)) {
                WriteError(res, 400, parse_err);
                return;
            }
            doc_ptr = &doc;
        }

        const std::string database = ReadStringField(req, doc_ptr, {"database", "name"});
        const std::string filename = ReadStringField(req, doc_ptr, {"filename", "file_name", "file"});

        std::string upload_id;
        std::string err;
        if (!upload_registry.InitSession(database, filename, upload_id, err)) {
            WriteError(res, 400, err);
            return;
        }

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("upload_id");
        writer.String(upload_id.c_str());
        writer.Key("message");
        writer.String("Upload session initialized.");
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Post("/apex/databases/upload/chunk", [&](const httplib::Request& req, httplib::Response& res) {
        if (create_tracker.IsRunning()) {
            WriteError(res, 423, "Cannot upload while database creation is running.");
            return;
        }

        std::string upload_id;
        uint64_t chunk_index = 0;
        std::string chunk_content;

        if (req.is_multipart_form_data()) {
            if (req.form.has_field("upload_id"))
                upload_id = req.form.get_field("upload_id");

            std::string index_str;
            if (req.form.has_field("chunk_index"))
                index_str = req.form.get_field("chunk_index");
            if (!ParseUInt64(index_str, &chunk_index)) {
                WriteError(res, 400, "chunk_index must be an unsigned integer.");
                return;
            }

            if (!req.form.has_file("data_chunk")) {
                WriteError(res, 400, "Missing upload chunk file field 'data_chunk'.");
                return;
            }
            auto chunk_file = req.form.get_file("data_chunk");
            chunk_content = chunk_file.content;
        } else {
            rapidjson::Document doc;
            std::string parse_err;
            const rapidjson::Document* doc_ptr = nullptr;
            if (!req.body.empty()) {
                if (!ParseJsonObject(req, doc, parse_err)) {
                    WriteError(res, 400, parse_err);
                    return;
                }
                doc_ptr = &doc;
            }
            upload_id = ReadStringField(req, doc_ptr, {"upload_id"});
            std::string chunk_index_str = ReadStringField(req, doc_ptr, {"chunk_index"});
            if (!ParseUInt64(chunk_index_str, &chunk_index)) {
                WriteError(res, 400, "chunk_index must be an unsigned integer.");
                return;
            }
            if (doc_ptr != nullptr && doc_ptr->HasMember("chunk_data") && (*doc_ptr)["chunk_data"].IsString())
                chunk_content = (*doc_ptr)["chunk_data"].GetString();
        }

        uint64_t total_bytes = 0;
        std::string err;
        if (!upload_registry.AppendChunk(upload_id, chunk_index, chunk_content, &total_bytes, err)) {
            WriteError(res, 400, err);
            return;
        }

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("upload_id");
        writer.String(upload_id.c_str());
        writer.Key("chunk_index");
        writer.Uint64(chunk_index);
        writer.Key("received_bytes");
        writer.Uint64(total_bytes);
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Get("/apex/databases/create/progress", [&](const httplib::Request&, httplib::Response& res) {
        const auto snapshot = create_tracker.GetSnapshot();

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("running");
        writer.Bool(snapshot.running);
        writer.Key("cancel_requested");
        writer.Bool(snapshot.cancel_requested);
        writer.Key("status");
        writer.String(snapshot.status.c_str());
        writer.Key("database");
        if (snapshot.database.empty())
            writer.Null();
        else
            writer.String(snapshot.database.c_str());
        writer.Key("progress_percent");
        writer.Int(snapshot.progress_percent);
        writer.Key("step");
        writer.String(snapshot.step.c_str());
        writer.Key("message");
        writer.String(snapshot.message.c_str());
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Post("/apex/databases/create/cancel", [&](const httplib::Request& req, httplib::Response& res) {
        rapidjson::Document doc;
        std::string parse_err;
        const rapidjson::Document* doc_ptr = nullptr;
        if (!req.body.empty()) {
            if (!ParseJsonObject(req, doc, parse_err)) {
                WriteError(res, 400, parse_err);
                return;
            }
            doc_ptr = &doc;
        }

        const std::string upload_id = ReadStringField(req, doc_ptr, {"upload_id"});
        if (!upload_id.empty())
            upload_registry.RemoveSession(upload_id);

        create_tracker.RequestCancel();
        const bool running = create_tracker.IsRunning();

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("running");
        writer.Bool(running);
        writer.Key("message");
        if (running)
            writer.String("Cancel request accepted. Waiting for current step to finish.");
        else
            writer.String("No running create task. Upload cache cleaned if provided.");
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Get("/apex/databases", [&](const httplib::Request&, httplib::Response& res) {
        if (reject_if_creating(res))
            return;

        const auto existing = manager.ExistingDatabases();
        const auto existing_details = manager.ExistingDatabaseDetails();
        const auto loaded = manager.LoadedDatabases();
        const auto details = manager.LoadedDatabaseDetails();
        const auto active = manager.Active();

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("active_database");
        if (active.exists)
            writer.String(active.name.c_str());
        else
            writer.Null();

        writer.Key("active_database_stats");
        if (active.exists)
            WriteDatabaseStats(writer, active.stats);
        else
            writer.Null();

        writer.Key("existing_databases");
        WriteDatabaseList(writer, existing);

        writer.Key("loaded_databases");
        WriteDatabaseList(writer, loaded);

        writer.Key("loaded_database_details");
        writer.StartArray();
        for (const auto& [name, stats] : details) {
            writer.StartObject();
            writer.Key("name");
            writer.String(name.c_str());
            writer.Key("stats");
            WriteDatabaseStats(writer, stats);
            writer.EndObject();
        }
        writer.EndArray();

        writer.Key("existing_database_details");
        writer.StartArray();
        for (const auto& [name, stats] : existing_details) {
            const DatabaseDiskUsage usage = ReadDatabaseDiskUsage(name);
            writer.StartObject();
            writer.Key("name");
            writer.String(name.c_str());
            writer.Key("stats");
            WriteDatabaseStats(writer, stats);
            writer.Key("index_size_bytes");
            writer.Uint64(usage.index_size_bytes);
            writer.Key("dictionary_size_bytes");
            writer.Uint64(usage.dictionary_size_bytes);
            writer.Key("total_size_bytes");
            writer.Uint64(usage.total_size_bytes);
            writer.Key("loaded");
            writer.Bool(std::find(loaded.begin(), loaded.end(), name) != loaded.end());
            writer.Key("active");
            writer.Bool(active.exists && active.name == name);
            writer.EndObject();
        }
        writer.EndArray();
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Post("/apex/databases/create", [&](const httplib::Request& req, httplib::Response& res) {
        std::string name;
        std::string data_file;
        std::string uploaded_temp_file;
        std::string upload_id;

        if (req.is_multipart_form_data()) {
            if (req.form.has_field("database"))
                name = req.form.get_field("database");
            if (name.empty() && req.form.has_field("name"))
                name = req.form.get_field("name");
            if (req.form.has_field("upload_id"))
                upload_id = req.form.get_field("upload_id");

            if (req.form.has_file("data_file")) {
                auto file = req.form.get_file("data_file");
                if (!HasNtExtension(file.filename)) {
                    WriteError(res, 400, "Only .nt files can be uploaded.");
                    return;
                }
                std::string upload_err;
                if (!SaveUploadedDataFile(name.empty() ? "db" : name, file, uploaded_temp_file, upload_err)) {
                    WriteError(res, 400, upload_err);
                    return;
                }
                data_file = uploaded_temp_file;
            }
        } else {
            rapidjson::Document doc;
            std::string parse_err;
            const rapidjson::Document* doc_ptr = nullptr;
            if (!req.body.empty()) {
                if (!ParseJsonObject(req, doc, parse_err)) {
                    WriteError(res, 400, parse_err);
                    return;
                }
                doc_ptr = &doc;
            }

            name = ReadStringField(req, doc_ptr, {"database", "name"});
            data_file = ReadStringField(req, doc_ptr, {"data_file", "file"});
            upload_id = ReadStringField(req, doc_ptr, {"upload_id"});
        }

        if (name.empty()) {
            WriteError(res, 400, "Database name is required.");
            return;
        }

        auto cleanup_uploaded_temp_files = [&]() {
            if (!uploaded_temp_file.empty()) {
                std::error_code ec;
                fs::remove(uploaded_temp_file, ec);
            }
            if (!upload_id.empty())
                upload_registry.RemoveSession(upload_id);
        };

        if (data_file.empty() && !upload_id.empty()) {
            std::string resolve_err;
            if (!upload_registry.ResolveFilePath(upload_id, data_file, resolve_err)) {
                cleanup_uploaded_temp_files();
                WriteError(res, 400, resolve_err);
                return;
            }
        }

        if (data_file.empty()) {
            cleanup_uploaded_temp_files();
            WriteError(res, 400, "Missing input file. Provide 'data_file' or 'upload_id'.");
            return;
        }

        std::string start_err;
        if (!create_tracker.TryStart(name, start_err)) {
            cleanup_uploaded_temp_files();
            WriteError(res, 409, start_err);
            return;
        }

        auto fail_create = [&](int status, const std::string& message, bool cancelled) {
            cleanup_uploaded_temp_files();
            if (cancelled)
                create_tracker.FinishCancelled(message);
            else
                create_tracker.FinishError(message);
            WriteError(res, status, message);
        };

        if (create_tracker.IsCancelRequested()) {
            fail_create(409, "Create request cancelled.", true);
            return;
        }

        create_tracker.Update(10, "校验NT格式", "正在检查文件内容。");
        uint64_t checked_lines = 0;
        std::string validate_err;
        if (!ValidateNtFile(data_file, validate_err, &checked_lines)) {
            fail_create(400, validate_err, false);
            return;
        }

        if (create_tracker.IsCancelRequested()) {
            fail_create(409, "Create request cancelled.", true);
            return;
        }

        create_tracker.Update(
            35, "准备建库", "文件校验通过，已检查前 " + std::to_string(checked_lines) + " 条 RDF 语句。");

        std::atomic<bool> build_done(false);
        std::thread build_progress_thread([&]() {
            int progress = 45;
            while (!build_done.load()) {
                create_tracker.Update(progress, "创建数据库", "正在构建索引与字典，请耐心等待。");
                progress = std::min(progress + 1, 85);
                std::this_thread::sleep_for(std::chrono::milliseconds(900));
            }
        });

        std::string err;
        const bool created = manager.CreateDatabase(name, data_file, err);
        build_done = true;
        build_progress_thread.join();

        if (!created) {
            std::string cleanup_err;
            if (!manager.RemoveDatabaseFiles(name, cleanup_err))
                err += " Cleanup failed: " + cleanup_err;
            fail_create(400, err, false);
            return;
        }

        if (create_tracker.IsCancelRequested()) {
            create_tracker.Update(90, "取消创建", "正在清理已生成数据。");
            std::string remove_err;
            if (!manager.RemoveDatabaseFiles(name, remove_err)) {
                fail_create(500, "Create cancelled, but cleanup failed: " + remove_err, true);
                return;
            }
            fail_create(409, "Create request cancelled.", true);
            return;
        }

        create_tracker.Update(92, "加载数据库", "正在激活新数据库。");
        DatabaseStats stats;
        if (!manager.LoadDatabase(name, err, &stats)) {
            fail_create(500, "Database created but failed to activate: " + err, false);
            return;
        }

        cleanup_uploaded_temp_files();
        create_tracker.FinishSuccess("Database created and activated: " + name);

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("message");
        writer.String(("Database created and activated: " + name).c_str());
        writer.Key("database");
        writer.String(name.c_str());
        writer.Key("stats");
        WriteDatabaseStats(writer, stats);
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Post("/apex/databases/load", [&](const httplib::Request& req, httplib::Response& res) {
        if (reject_if_creating(res))
            return;

        rapidjson::Document doc;
        std::string parse_err;
        const rapidjson::Document* doc_ptr = nullptr;
        if (!req.body.empty()) {
            if (!ParseJsonObject(req, doc, parse_err)) {
                WriteError(res, 400, parse_err);
                return;
            }
            doc_ptr = &doc;
        }

        const std::string name = ReadStringField(req, doc_ptr, {"database", "name"});
        std::string err;
        DatabaseStats stats;
        if (!manager.LoadDatabase(name, err, &stats)) {
            WriteError(res, 400, err);
            return;
        }

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("message");
        writer.String(("Database loaded and activated: " + name).c_str());
        writer.Key("database");
        writer.String(name.c_str());
        writer.Key("stats");
        WriteDatabaseStats(writer, stats);
        writer.Key("active_database");
        writer.String(name.c_str());
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Post("/apex/databases/delete", [&](const httplib::Request& req, httplib::Response& res) {
        if (reject_if_creating(res))
            return;

        rapidjson::Document doc;
        std::string parse_err;
        const rapidjson::Document* doc_ptr = nullptr;
        if (!req.body.empty()) {
            if (!ParseJsonObject(req, doc, parse_err)) {
                WriteError(res, 400, parse_err);
                return;
            }
            doc_ptr = &doc;
        }

        const std::string name = ReadStringField(req, doc_ptr, {"database", "name"});
        if (name.empty()) {
            WriteError(res, 400, "Database name is required.");
            return;
        }

        const auto existing = manager.ExistingDatabases();
        if (std::find(existing.begin(), existing.end(), name) == existing.end()) {
            WriteError(res, 404, "Database not found: " + name);
            return;
        }

        std::string err;
        if (!manager.RemoveDatabaseFiles(name, err)) {
            WriteError(res, 500, err);
            return;
        }

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("message");
        writer.String(("Database deleted: " + name).c_str());
        writer.Key("database");
        writer.String(name.c_str());
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Get("/apex/database", [&](const httplib::Request&, httplib::Response& res) {
        if (reject_if_creating(res))
            return;

        const auto active = manager.Active();
        const auto loaded = manager.LoadedDatabases();

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);

        writer.Key("active_database");
        if (active.exists)
            writer.String(active.name.c_str());
        else
            writer.Null();

        writer.Key("active_database_stats");
        if (active.exists)
            WriteDatabaseStats(writer, active.stats);
        else
            writer.Null();

        writer.Key("loaded_databases");
        WriteDatabaseList(writer, loaded);
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    server.Get("/apex/stats", [&](const httplib::Request&, httplib::Response& res) {
        if (reject_if_creating(res))
            return;

        const auto active = manager.Active();
        if (!active.exists) {
            WriteError(res, 400, "No active database. Please load and switch a database first.");
            return;
        }

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("ok");
        writer.Bool(true);
        writer.Key("database");
        writer.String(active.name.c_str());
        writer.Key("stats");
        WriteDatabaseStats(writer, active.stats);
        writer.EndObject();

        SetCorsHeaders(res);
        res.set_content(buf.GetString(), "application/json;charset=utf-8");
    });

    const auto query_handler = [&](const httplib::Request& req, httplib::Response& res) {
        if (reject_if_creating(res))
            return;

        rapidjson::Document doc;
        const rapidjson::Document* doc_ptr = nullptr;
        if (!req.body.empty()) {
            doc.Parse(req.body.c_str());
            if (!doc.HasParseError() && doc.IsObject())
                doc_ptr = &doc;
        }

        std::string sparql = ReadStringField(req, doc_ptr, {"query"});
        if (sparql.empty() && doc_ptr == nullptr && !req.body.empty())
            sparql = req.body;

        const std::string db_name = ReadStringField(req, doc_ptr, {"database", "db"});
        if (!db_name.empty()) {
            std::string err;
            if (!manager.LoadDatabase(db_name, err)) {
                WriteError(res, 400, err);
                return;
            }
        }

        const auto active = manager.Active();
        if (!active.exists || !active.index) {
            WriteError(res, 400, "No active database. Please load and switch a database first.");
            return;
        }
        if (sparql.empty()) {
            WriteError(res, 400, "Missing SPARQL query. Use query parameter or JSON field 'query'.");
            return;
        }

        try {
            const auto begin = std::chrono::high_resolution_clock::now();

            SPARQLParser parser(sparql);
            QueryExecutor executor(active.index, parser, max_threads);
            executor.Query();
            const auto& project_vars = parser.ProjectVariables();

            rapidjson::StringBuffer buf;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
            writer.StartObject();
            writer.Key("ok");
            writer.Bool(true);
            writer.Key("database");
            writer.String(active.name.c_str());
            writer.Key("database_stats");
            WriteDatabaseStats(writer, active.stats);

            writer.Key("head");
            writer.StartObject();
            writer.Key("vars");
            writer.StartArray();
            for (const auto& var : project_vars)
                writer.String(var.c_str());
            writer.EndArray();
            writer.EndObject();

            writer.Key("results");
            writer.StartObject();
            writer.Key("bindings");
            writer.StartArray();
            const QueryExecutor::ResultStreamStats stream_stats = executor.StreamResultRows(
                [&writer](const std::vector<std::string>& row) {
                    writer.StartArray();
                    for (const auto& value : row)
                        writer.String(value.c_str());
                    writer.EndArray();
                },
                kMaxQueryResultRows);
            writer.EndArray();
            writer.Key("count");
            writer.Uint64(stream_stats.row_count);
            writer.Key("returned_count");
            writer.Uint64(stream_stats.returned_row_count);
            writer.Key("truncated");
            writer.Bool(stream_stats.truncated);
            writer.Key("truncated_limit");
            writer.Uint64(kMaxQueryResultRows);
            writer.EndObject();

            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> diff = end - begin;
            const uint64_t row_count = stream_stats.row_count;
            const uint64_t returned_row_count = stream_stats.returned_row_count;
            const uint64_t column_count = project_vars.size();
            const uint64_t cell_count = returned_row_count * column_count;
            const uint64_t total_cell_count = row_count * column_count;
            const double non_empty_ratio =
                cell_count == 0 ? 0.0 : (100.0 * stream_stats.non_empty_cells / cell_count);
            const double avg_value_chars = cell_count == 0 ? 0.0 : (1.0 * stream_stats.total_chars / cell_count);

            writer.Key("metrics");
            writer.StartObject();
            writer.Key("elapsed_ms");
            writer.Double(diff.count());
            writer.Key("execute_ms");
            writer.Double(executor.execute_cost());
            writer.Key("build_group_ms");
            writer.Double(executor.build_group_cost());
            writer.Key("gen_result_ms");
            writer.Double(executor.gen_result_cost());
            writer.EndObject();

            writer.Key("query_stats");
            writer.StartObject();
            writer.Key("row_count");
            writer.Uint64(row_count);
            writer.Key("returned_row_count");
            writer.Uint64(returned_row_count);
            writer.Key("column_count");
            writer.Uint64(column_count);
            writer.Key("cell_count");
            writer.Uint64(cell_count);
            writer.Key("total_cell_count");
            writer.Uint64(total_cell_count);
            writer.Key("non_empty_cells");
            writer.Uint64(stream_stats.non_empty_cells);
            writer.Key("non_empty_ratio_percent");
            writer.Double(non_empty_ratio);
            writer.Key("avg_value_chars");
            writer.Double(avg_value_chars);
            writer.Key("truncated");
            writer.Bool(stream_stats.truncated);
            writer.Key("truncated_limit");
            writer.Uint64(kMaxQueryResultRows);
            writer.EndObject();

            writer.EndObject();

            SetCorsHeaders(res);
            res.set_content(buf.GetString(), "application/sparql-results+json;charset=utf-8");
        } catch (const SPARQLParser::ParserException& e) {
            WriteError(res, 400, e.to_string());
        } catch (const std::exception& e) {
            WriteError(res, 500, e.what());
        }
    };

    server.Get("/apex/query", query_handler);
    server.Post("/apex/query", query_handler);

    // Alias for compatibility with SPARQL naming.
    server.Get("/apex/sparql", query_handler);
    server.Post("/apex/sparql", query_handler);

    server.Options("/apex/databases", options_handler);
    server.Options("/apex/databases/upload/init", options_handler);
    server.Options("/apex/databases/upload/chunk", options_handler);
    server.Options("/apex/databases/create", options_handler);
    server.Options("/apex/databases/create/progress", options_handler);
    server.Options("/apex/databases/create/cancel", options_handler);
    server.Options("/apex/databases/load", options_handler);
    server.Options("/apex/databases/delete", options_handler);
    server.Options("/apex/database", options_handler);
    server.Options("/apex/stats", options_handler);
    server.Options("/apex/query", options_handler);
    server.Options("/apex/sparql", options_handler);

    server.Post("/apex/disconnect", [&](const httplib::Request&, httplib::Response& res) {
        SetCorsHeaders(res);
        res.set_content("{\"ok\":true,\"message\":\"Disconnected\"}", "application/json;charset=utf-8");
        server.stop();
    });

    std::cout << "APEX server listening on " << host << ":" << port << std::endl;
    std::cout << "UI: http://" << host << ":" << port << "/apex/ui/" << std::endl;
    std::cout << "Max upload payload: " << max_upload_mb << " MB" << std::endl;
    std::cout << "No database is loaded at startup. Use /apex/databases/* APIs." << std::endl;
    if (!server.listen(host, port)) {
        std::cerr << "failed to start server on " << host << ":" << port << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace apex
