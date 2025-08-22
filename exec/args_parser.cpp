#include "exec/args_parser.hpp"

void ArgsParser::Build(const std::unordered_map<std::string, std::string>& args) {
    if (args.empty() || args.count("-h") || args.count("--help")) {
        std::cout << help_info_ << std::endl;
        exit(1);
    }
    if ((!args.count("-d") && !args.count("--database")) || (!args.count("-f") && !args.count("--file"))) {
        std::cerr << "usage: epei build [-d DATABASE] [-f FILE]" << std::endl;
        std::cerr << "epei: error: the following arguments are required: [-d DATABASE] [-f FILE]" << std::endl;
        exit(1);
    }
    arguments_[arg_db_path_] = args.count("-d") ? args.at("-d") : args.at("--database");
    arguments_[arg_file_] = args.count("-f") ? args.at("-f") : args.at("--file");
}

void ArgsParser::Query(const std::unordered_map<std::string, std::string>& args) {
    if (args.count("-h") || args.count("--help")) {
        std::cout << help_info_ << std::endl;
        exit(1);
    }

    if (args.count("-d"))
        arguments_[arg_db_path_] = args.count("-d") ? args.at("-d") : args.at("--database");
    if (args.count("-f"))
        arguments_[arg_file_] = args.at("-f");
    else if (args.count("--file"))
        arguments_[arg_file_] = args.at("--file");
    else
        arguments_[arg_file_] = "";
}

void ArgsParser::Train(const std::unordered_map<std::string, std::string>& args) {
    if (args.count("-h") || args.count("--help")) {
        std::cout << help_info_ << std::endl;
        exit(1);
    }

    if (args.count("-d"))
        arguments_[arg_db_path_] = args.count("-d") ? args.at("-d") : args.at("--database");

    if (args.count("-f"))
        arguments_[arg_file_] = args.at("-f");
    else if (args.count("--file"))
        arguments_[arg_file_] = args.at("--file");
    else
        arguments_[arg_file_] = "";
}

ArgsParser::CommandT ArgsParser::Parse(int argc, char** argv) {
    if (argc == 1) {
        std::cout << help_info_ << std::endl;
        std::cerr << "error: the following arguments are required: command" << std::endl;
        exit(1);
    }

    std::string argv1 = std::string(argv[1]);

    if (argc == 2 && (argv1 == "-h" || argv1 == "--help")) {
        std::cout << help_info_ << std::endl;
        exit(1);
    }

    if (!position_.count(argv1)) {
        std::cout << help_info_ << std::endl;
        std::cerr << "error: the following arguments are required: command" << std::endl;
        exit(1);
    }

    std::unordered_map<std::string, std::string> args;
    for (int i = 2; i < argc; i += 2) {
        if (argv[i][0] != '-') {
            std::cerr << "error: unrecognized arguments: " << argv[i] << std::endl;
            exit(1);
        }
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            args.emplace(argv[i], "");
            i = i - 1;  // because when it enters the next loop, the i will be plus 2,
            // so we need to decrease one in order to ensure the rest flags and arguments are one-to-one
            // correspondence
            continue;
        }
        if (i + 1 >= argc || argv[i + 1][0] == '-') {
            std::cerr << "error: argument " << argv[i] << ": expected one argument" << std::endl;
            exit(1);
        }
        args.emplace(argv[i], argv[i + 1]);
    }

    // 执行对应的命令的解析器
    (this->*selector_[argv1])(args);
    return position_[argv1];
}

const std::unordered_map<std::string, std::string>& ArgsParser::Arguments() const {
    return arguments_;
}
