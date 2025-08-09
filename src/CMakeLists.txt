set(this avpjoin_lib)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
        avpjoin.cpp
        server/*.cpp
        parser/*.cpp
        index/*.cpp
        dictionary/*.cpp
        query/*.cpp
        utils/*.cpp)

add_library(${this} STATIC ${srcs})
target_link_libraries(${this} PRIVATE phmap httplib streamvbyte)
set_target_properties(${this} PROPERTIES LINKER_LANGUAGE CXX)
