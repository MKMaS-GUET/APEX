set(this apex)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
        *.hpp *.cpp)

add_executable(${this} ${srcs})
target_link_libraries(${this} apex_lib)
