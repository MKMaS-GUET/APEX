set(this apex_lib)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
        dictionary/*.cpp
        exec/*.cpp
        index/*.cpp
        parser/*.cpp
        query/*.cpp
        utils/*.cpp)

add_library(${this} STATIC ${srcs})
# phmap is header-only and required by public headers (e.g., index_builder.hpp),
# so expose it transitively to dependents.
target_link_libraries(${this} PUBLIC phmap PRIVATE streamvbyte TBB::tbb)
set_target_properties(${this} PROPERTIES LINKER_LANGUAGE CXX)

add_executable(apex exec/main.cpp)
target_link_libraries(apex PRIVATE apex_lib ${SYSTEM_STDCXX_LIBRARY} ${SYSTEM_GCC_S_LIBRARY})

find_package(Threads REQUIRED)
add_executable(apex_server interface/server.cpp)
target_link_libraries(
        apex_server
        PRIVATE
        apex_lib
        httplib::httplib
        Threads::Threads
        ${SYSTEM_STDCXX_LIBRARY}
        ${SYSTEM_GCC_S_LIBRARY})
