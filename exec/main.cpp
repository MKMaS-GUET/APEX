#include <avpjoin/avpjoin.hpp>

#include "exec/args_parser.hpp"

void Build(const std::unordered_map<std::string, std::string>& arguments) {
    std::string db_name = arguments.at("path");
    std::string data_file = arguments.at("file");
    avpjoin::AVPJoin::Create(db_name, data_file);
}

void Query(const std::unordered_map<std::string, std::string>& arguments) {
    std::string db_path;
    std::string query_path;
    if (arguments.count("path")) {
        db_path = arguments.at("path");
        if (db_path.find("/") == std::string::npos)
            db_path = "./DB_DATA_ARCHIVE/" + db_path;
    }

    if (arguments.count("file"))
        query_path = arguments.at("file");

    avpjoin::AVPJoin::Query(db_path, query_path);
}

void Train(const std::unordered_map<std::string, std::string>& arguments) {
    std::string db_path;
    std::string query_path;
    if (arguments.count("path")) {
        db_path = arguments.at("path");
        if (db_path.find("/") == std::string::npos)
            db_path = "./DB_DATA_ARCHIVE/" + db_path;
    }

    if (arguments.count("file"))
        query_path = arguments.at("file");

    avpjoin::AVPJoin::Train(db_path, query_path);
}

void Test(const std::unordered_map<std::string, std::string>& arguments) {
    std::string db_path;
    std::string query_path;
    if (arguments.count("path")) {
        db_path = arguments.at("path");
        if (db_path.find("/") == std::string::npos)
            db_path = "./DB_DATA_ARCHIVE/" + db_path;
    }
    if (arguments.count("file"))
        query_path = arguments.at("file");

    avpjoin::AVPJoin::Test(db_path, query_path);
}

struct EnumClassHash {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

std::unordered_map<ArgsParser::CommandT, void (*)(const std::unordered_map<std::string, std::string>&), EnumClassHash>
    selector;

int main(int argc, char** argv) {
    selector = {{ArgsParser::CommandT::kBuild, &Build},
                {ArgsParser::CommandT::kQuery, &Query},
                {ArgsParser::CommandT::kTrain, &Train},
                {ArgsParser::CommandT::kTest, &Test}};

    auto parser = ArgsParser();
    auto command = parser.Parse(argc, argv);
    auto arguments = parser.Arguments();
    selector[command](arguments);
    return 0;
}
