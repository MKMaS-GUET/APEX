#include "apex.cpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {
constexpr std::string_view kDbArchive = "./DB_DATA_ARCHIVE/";
constexpr std::string_view kHelpInfo =
    "Usage: apex [COMMAND] [OPTIONS]\n"
    "\n"
    "Commands:\n"
    "  build      Build an RDF database.\n"
    "  query      Query an RDF database.\n"
    "  train      Train on an database.\n"
    "  test       Test on an database.\n"
    "\n"
    "Options:\n"
    "  -h, --help  Show this help message and exit.\n"
    "\n"
    "Commands:\n"
    "\n"
    "  build\n"
    "    Build an RDF database.\n"
    "\n"
    "    Usage: apex build [OPTIONS]\n"
    "\n"
    "    Options:\n"
    "      -d, --database <PATH>   Specify the path of the database.\n"
    "      -f, --file <FILE>       Specify the input file to build the database.\n"
    "\n"
    "  query\n"
    "    Query an RDF database.\n"
    "\n"
    "    Usage: apex query [OPTIONS]\n"
    "\n"
    "    Options:\n"
    "      -d, --database <PATH>   Specify the path of the database.\n"
    "      -f, --file <FILE>       Specify the file containing the queries.\n"
    "\n"
    "  train\n"
    "    Train variable order generator.\n"
    "\n"
    "    Usage: apex train [OPTIONS]\n"
    "\n"
    "    Options:\n"
    "      -d, --database <NAME>   Specify the name of the database.\n"
    "      -f, --file <FILE>       Specify the file containing the queries to train.\n"
    "\n"
    "  test\n"
    "    Test an RDF database.\n"
    "\n"
    "    Usage: apex test [OPTIONS]\n"
    "\n"
    "    Options:\n"
    "      -d, --database <NAME>   Specify the name of the database.\n"
    "      -f, --file <FILE>       Specify the file containing the queries to test.\n";

[[noreturn]] void PrintHelpAndExit(int code) {
    std::cout << kHelpInfo << std::endl;
    std::exit(code);
}

std::string NormalizeDbPath(const std::string& db_path_raw) {
    if (db_path_raw.empty())
        return {};
    std::string db_path = db_path_raw;
    if (db_path.find('/') == std::string::npos)
        db_path = std::string(kDbArchive) + db_path;
    return db_path;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << kHelpInfo << std::endl;
        std::cerr << "error: the following arguments are required: command" << std::endl;
        return 0;
    }

    const std::string command = argv[1];
    if (command == "-h" || command == "--help") {
        PrintHelpAndExit(0);
    }
    if (command != "build" && command != "query" && command != "train" && command != "test") {
        std::cout << kHelpInfo << std::endl;
        std::cerr << "error: the following arguments are required: command" << std::endl;
        return 0;
    }

    std::string db_path;
    std::string file_path;
    auto consume_value = [&](int& i, const std::string& flag) -> std::string {
        if (i + 1 >= argc || argv[i + 1][0] == '-') {
            std::cerr << "error: argument " << flag << ": expected one argument" << std::endl;
            std::exit(1);
        }
        return argv[++i];
    };

    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "-h" || flag == "--help") {
            PrintHelpAndExit(0);
        }
        if (flag == "-d" || flag == "--database") {
            db_path = consume_value(i, flag);
            continue;
        }
        if (flag == "-f" || flag == "--file") {
            file_path = consume_value(i, flag);
            continue;
        }

        std::cerr << "error: unrecognized arguments: " << flag << std::endl;
        return 0;
    }

    auto validate_args = [&]() -> bool {
        if (db_path.empty() || file_path.empty()) {
            std::cerr << "usage: apex " << command << " [-d DATABASE] [-f FILE]" << std::endl;
            if (command == "build")
                std::cerr << "apex: error: the following arguments are required: [-d DATABASE] [-f DATA FILE]"
                          << std::endl;
            else
                std::cerr << "apex: error: the following arguments are required: [-d DATABASE] [-f QUERY FILE]"
                          << std::endl;
            return false;
        }
        return true;
    };

    if (command == "build") {
        if (validate_args())
            apex::Create(db_path, file_path);
    } else if (command == "query") {
        if (validate_args())
            apex::Query(NormalizeDbPath(db_path), file_path);
    } else if (command == "train") {
        if (validate_args())
            apex::Train(NormalizeDbPath(db_path), file_path);
    } else if (command == "test") {
        if (validate_args())
            apex::Test(NormalizeDbPath(db_path), file_path);
    }
    return 0;
}
