set(this apex_lib)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
        apex.cpp
        server/*.cpp
        parser/*.cpp
        index/*.cpp
        dictionary/*.cpp
        query/*.cpp
        utils/*.cpp)

add_library(${this} STATIC ${srcs})
# phmap is header-only and required by public headers (e.g., index_builder.hpp),
# so expose it transitively to dependents.
target_link_libraries(${this} PUBLIC phmap PRIVATE httplib streamvbyte TBB::tbb)
set_target_properties(${this} PROPERTIES LINKER_LANGUAGE CXX)

add_executable(apex exec/main.cpp)
target_link_libraries(apex apex_lib)
